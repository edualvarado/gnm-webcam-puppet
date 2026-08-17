# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Solves GNM parameters from tracked face landmarks.

Each frame is resolved in two stages that separate rigid from non-rigid motion:

  1. A similarity transform is fit from the skull-fixed landmark subset onto the
     *neutral landmark reference* recorded during calibration. Applying it moves
     the observation into model space and removes head rotation, translation and
     scale; the rotation that was removed *is* the head pose.
  2. With rigid motion gone, the landmarks' displacement from the neutral
     reference is pure expression, so the expression coefficients follow from a
     regularized linear least-squares projection onto the GNM expression basis.

Tracking is deliberately *differential*: stage 2 fits how far landmarks have
moved from their neutral positions, not where they are in absolute terms. The
landmarker reports its own face shape with a non-metric z, so its landmarks
never coincide with the GNM vertices they map to; fitting absolute positions
makes a resting face register millimetres of invented expression, while fitting
displacements cancels that constant offset.

Stage 2 reuses `gnm.shape.fitting_utils.project_on_pca.PCABasisProjection`,
which pre-factors the basis so per-frame solves are a single matrix product.
"""

from collections.abc import Mapping
import dataclasses
import re

from gnm.shape import gnm_numpy
from gnm.shape.fitting_utils import project_on_pca
import immutabledict
import numpy as np
import numpy.typing as npt
from scipy.spatial import transform as scipy_transform

from webcam_puppet import correspondence as correspondence_lib
from webcam_puppet import geometry

FloatArray = npt.NDArray[np.floating]
IntArray = npt.NDArray[np.integer]

# Re-exported so callers and tests have one import site for the solver's maths.
landmarks_to_model_axes = geometry.landmarks_to_model_axes
fit_similarity_transform = geometry.fit_similarity_transform

# GNM's head joint carries 99.7% of the skinning weight over the face region,
# so head pose belongs there rather than on the neck.
_HEAD_JOINT_NAME = 'head'

# The expression basis is concatenated per face region, not ordered by global
# variance: components 0-99 are the left eye, 100-199 the right eye, 200-349 the
# lower face, 350-381 the tongue and 382 the pupils. Taking the leading N
# components across the whole basis would therefore solve for eyelids only and
# leave the mouth rigid, so components are budgeted per region instead.
# Within a region they are variance-ordered (energy falls ~100x from first to
# last), which makes truncating inside a region meaningful.
#
# Tongue and pupil components are omitted: neither is observable from outside
# the face, so fitting them would only absorb landmark noise.
_DEFAULT_REGION_COMPONENTS = immutabledict.immutabledict({
    'left_eye_region': 40,
    'right_eye_region': 40,
    'lower_face_region': 80,
})

# Tuned by sweeping recovery of known expressions from synthetic landmarks (see
# solver_test.py). The eye and lower-face basis blocks differ ~100x in
# displacement energy, so a ridge heavy enough to tame the mouth silently
# flattens eyelid motion: at 2e-4 a known eye expression was reconstructed with
# more error than simply predicting neutral. 1e-5 recovers ~87% of mouth motion
# and stays stable under 1mm of landmark noise.
_DEFAULT_EXPRESSION_REGULARIZATION = 1e-5
_DEFAULT_IDENTITY_COMPONENTS = 40
_DEFAULT_IDENTITY_REGULARIZATION = 2e-3


def expression_region_blocks(gnm: gnm_numpy.GNM) -> dict[str, IntArray]:
  """Groups expression component indices by face region.

  Region membership is read from the component names rather than hard-coded
  offsets, so a model revision that resizes a region stays supported.

  Args:
    gnm: The GNM model.

  Returns:
    A mapping from region name to that region's component indices, in order.
  """
  blocks: dict[str, list[int]] = {}
  for index, name in enumerate(gnm.expression_names):
    region = re.sub(r'_(?:\d+|mean)$', '', str(name))
    blocks.setdefault(region, []).append(index)
  return {
      region: np.asarray(indices, dtype=np.int32)
      for region, indices in blocks.items()
  }


def select_expression_components(
    gnm: gnm_numpy.GNM,
    region_components: Mapping[str, int] = _DEFAULT_REGION_COMPONENTS,
) -> IntArray:
  """Selects expression components to solve for, budgeted per region.

  Args:
    gnm: The GNM model.
    region_components: How many leading components to take from each named
      region. Regions absent from the mapping are not solved for.

  Returns:
    Sorted component indices into the expression basis, (K,).

  Raises:
    ValueError: If a requested region does not exist, or a budget is negative.
  """
  blocks = expression_region_blocks(gnm)
  unknown = set(region_components) - set(blocks)
  if unknown:
    raise ValueError(
        f'Unknown expression region(s) {sorted(unknown)}; available regions are'
        f' {sorted(blocks)}.'
    )

  selected: list[IntArray] = []
  for region, count in region_components.items():
    if count < 0:
      raise ValueError(
          f'Component budget for {region} must be >= 0, got {count}.'
      )
    selected.append(blocks[region][:count])

  if not selected or sum(len(block) for block in selected) == 0:
    raise ValueError('No expression components selected.')
  return np.sort(np.concatenate(selected))


@dataclasses.dataclass(frozen=True)
class FaceParameters:
  """GNM parameters recovered from one frame.

  Attributes:
    identity: Identity coefficients, (identity_dim,).
    expression: Expression coefficients, (expression_dim,).
    rotations: Per-joint axis-angle rotations, (num_joints, 3).
    translation: Global translation, (3,).
    landmark_rmse: RMS distance in metres between the aligned landmark targets
      and the fitted mesh at the corresponded vertices. A useful
      tracking-quality readout; roughly 1-3mm indicates a good fit.
  """

  identity: FloatArray
  expression: FloatArray
  rotations: FloatArray
  translation: FloatArray
  landmark_rmse: float


class LandmarkSolver:
  """Recovers GNM parameters from tracked landmarks.

  The expression basis is pre-factored once at construction, so each frame costs
  one similarity fit plus one matrix product.
  """

  def __init__(
      self,
      gnm: gnm_numpy.GNM,
      correspondence: correspondence_lib.Correspondence,
      region_components: Mapping[str, int] = _DEFAULT_REGION_COMPONENTS,
      expression_regularization: float = _DEFAULT_EXPRESSION_REGULARIZATION,
      identity_components: int = _DEFAULT_IDENTITY_COMPONENTS,
      identity_regularization: float = _DEFAULT_IDENTITY_REGULARIZATION,
      smoothing: float = 0.4,
      expression_gain: float = 1.0,
  ):
    """Builds the solver and pre-factors the bases.

    Args:
      gnm: The GNM model to drive.
      correspondence: Landmark-to-vertex mapping from `correspondence`.
      region_components: How many leading expression components to solve for per
        face region. More components track finer motion but amplify landmark
        noise. See `select_expression_components`.
      expression_regularization: Ridge weight on the expression coefficients.
      identity_components: Leading identity components used by
        `solve_identity`. Unlike the expression basis, the identity basis is not
        region-blocked, so a leading count is meaningful here.
      identity_regularization: Ridge weight on the identity coefficients.
      smoothing: Exponential-moving-average factor in [0, 1) applied to the
        solved parameters. 0 disables smoothing; higher is smoother but laggier.
      expression_gain: Multiplier applied to the solved expression coefficients.

        This is deliberate exaggeration, not accuracy: it does not improve the
        fit, and `landmark_rmse` is reported for the unscaled solution. Landmark
        tracking recovers the direction of expression well but only ~25% of its
        magnitude, so a gain near 3 makes motion read at natural strength on
        screen. Leave it at 1.0 when the coefficients are used quantitatively.

    Raises:
      ValueError: If smoothing is outside [0, 1), or expression_gain is not
        positive.
    """
    if not 0.0 <= smoothing < 1.0:
      raise ValueError(f'smoothing must be in [0, 1), got {smoothing}.')
    if expression_gain <= 0.0:
      raise ValueError(
          f'expression_gain must be positive, got {expression_gain}.'
      )

    self._gnm = gnm
    self._correspondence = correspondence
    self._smoothing = smoothing
    self._expression_gain = expression_gain
    self._expression_dim = gnm.expression_dim
    self._identity_dim = gnm.identity_dim

    self._vertex_indices = correspondence.vertex_indices
    self._rigid_vertex_indices = correspondence.vertex_indices[
        correspondence.rigid
    ]

    self._neutral_vertices = np.asarray(
        gnm.template_vertex_positions, dtype=np.float32
    )
    self._head_joint = list(gnm.joint_names).index(_HEAD_JOINT_NAME)

    # Solve only the selected components by handing the projection a basis that
    # already contains just those, then scattering coefficients back to full
    # length afterwards.
    self._expression_components = select_expression_components(
        gnm, region_components
    )
    self._expression_subbasis = np.ascontiguousarray(
        gnm.expression_basis[self._expression_components]
    )
    self._expression_projection = project_on_pca.PCABasisProjection(
        mean_vertex_positions=self._neutral_vertices,
        vertex_basis=self._expression_subbasis,
        vertex_indices=self._vertex_indices,
        regularization=expression_regularization,
        compute_reconstruction=False,
    )
    self._identity_projection = project_on_pca.PCABasisProjection(
        mean_vertex_positions=self._neutral_vertices,
        vertex_basis=gnm.vertex_identity_basis,
        vertex_indices=self._vertex_indices,
        regularization=identity_regularization,
        num_components=identity_components,
        compute_reconstruction=False,
    )

    self._identity = np.zeros(self._identity_dim, dtype=np.float32)
    self._smoothed: FaceParameters | None = None

  @property
  def identity(self) -> FloatArray:
    """The identity coefficients currently applied."""
    return self._identity

  def reset(self) -> None:
    """Clears identity and temporal smoothing state."""
    self._identity = np.zeros(self._identity_dim, dtype=np.float32)
    self._smoothed = None

  def _landmark_displacement(
      self,
      landmarks: FloatArray,
      image_width: int,
      image_height: int,
  ) -> tuple[FloatArray, FloatArray]:
    """Measures landmark displacement from neutral, and the head pose.

    Aligns the observation onto the neutral landmark reference rather than onto
    the GNM mesh: matching like with like keeps the landmarker's shape bias out
    of the residual, leaving only genuine motion.

    Args:
      landmarks: MediaPipe landmarks, (M_all, 3).
      image_width: Frame width in pixels.
      image_height: Frame height in pixels.

    Returns:
      A tuple of per-landmark displacement in metres, (M, 3), and the head's
      axis-angle rotation, (3,).
    """
    observed = geometry.landmarks_to_model_axes(
        landmarks, image_width, image_height
    )
    observed = observed[self._correspondence.landmark_indices]
    reference = self._correspondence.reference_landmarks

    scale, rotation, translation = geometry.fit_similarity_transform(
        observed[self._correspondence.rigid],
        reference[self._correspondence.rigid],
    )
    aligned = geometry.apply_similarity_transform(
        observed, scale, rotation, translation
    )

    # The fit maps the observation into the neutral frame, so the head's own
    # rotation is its inverse.
    head_rotation = scipy_transform.Rotation.from_matrix(rotation.T).as_rotvec()
    return (aligned - reference).astype(np.float32), head_rotation.astype(
        np.float32
    )

  def _scatter_displacement(self, displacement: FloatArray) -> FloatArray:
    """Turns per-landmark displacement into a full-length vertex target array.

    `PCABasisProjection` subtracts the neutral mesh and indexes the result, so
    adding the displacement onto a copy of that mesh makes the projection see
    exactly the displacement at the corresponded rows and zero elsewhere. Only
    the corresponded rows are ever read.

    Args:
      displacement: Landmark displacement from neutral in metres, (M, 3).

    Returns:
      A full vertex array carrying the displacement, (V, 3).
    """
    targets = self._neutral_vertices.copy()
    targets[self._vertex_indices] += displacement
    return targets

  def solve_identity(
      self,
      landmarks: FloatArray,
      image_width: int,
      image_height: int,
  ) -> FloatArray:
    """Estimates and stores identity from a neutral-expression frame.

    Call this once while the subject holds a neutral face. Identity is then held
    fixed for subsequent `solve` calls, which keeps per-frame solving to
    expression and pose only.

    Because tracking is differential, identity is fit to how the subject's
    neutral face differs from the calibration reference, which is the part of
    their shape the landmarker actually resolves.

    Args:
      landmarks: MediaPipe landmarks from a neutral frame, (M_all, 3).
      image_width: Frame width in pixels.
      image_height: Frame height in pixels.

    Returns:
      The identity coefficients, (identity_dim,).
    """
    displacement, _ = self._landmark_displacement(
        landmarks, image_width, image_height
    )
    result = self._identity_projection(
        self._scatter_displacement(displacement)
    )

    identity = np.zeros(self._identity_dim, dtype=np.float32)
    coefficients = np.asarray(result.coefficients)[0]
    identity[: len(coefficients)] = coefficients
    self._identity = identity
    return identity

  def solve(
      self,
      landmarks: FloatArray,
      image_width: int,
      image_height: int,
  ) -> FaceParameters:
    """Solves expression and head pose for one frame.

    Args:
      landmarks: MediaPipe landmarks, (M_all, 3).
      image_width: Frame width in pixels.
      image_height: Frame height in pixels.

    Returns:
      The parameters for this frame, temporally smoothed.
    """
    displacement, head_rotation = self._landmark_displacement(
        landmarks, image_width, image_height
    )

    result = self._expression_projection(
        self._scatter_displacement(displacement)
    )

    coefficients = np.asarray(result.coefficients)[0]
    expression = np.zeros(self._expression_dim, dtype=np.float32)
    expression[self._expression_components] = (
        coefficients * self._expression_gain
    )

    rotations = np.zeros((self._gnm.num_joints, 3), dtype=np.float32)
    rotations[self._head_joint] = head_rotation

    # How much of the observed displacement the fitted expression explains.
    # Computed from the unscaled coefficients so the readout stays a fit
    # diagnostic rather than reflecting any exaggeration gain.
    residual = (
        np.einsum(
            'i,ivm->vm',
            coefficients,
            self._expression_subbasis[:, self._vertex_indices],
        )
        - displacement
    )
    landmark_rmse = float(
        np.sqrt(np.mean(np.sum(np.square(residual), axis=-1)))
    )

    parameters = FaceParameters(
        identity=self._identity,
        expression=expression,
        rotations=rotations,
        translation=np.zeros(3, dtype=np.float32),
        landmark_rmse=landmark_rmse,
    )
    return self._smooth(parameters)

  def _smooth(self, parameters: FaceParameters) -> FaceParameters:
    """Blends parameters with the previous frame's."""
    if self._smoothing == 0.0 or self._smoothed is None:
      self._smoothed = parameters
      return parameters

    weight = self._smoothing
    blended = FaceParameters(
        identity=parameters.identity,
        expression=(
            weight * self._smoothed.expression
            + (1.0 - weight) * parameters.expression
        ),
        rotations=(
            weight * self._smoothed.rotations
            + (1.0 - weight) * parameters.rotations
        ),
        translation=parameters.translation,
        landmark_rmse=parameters.landmark_rmse,
    )
    self._smoothed = blended
    return blended
