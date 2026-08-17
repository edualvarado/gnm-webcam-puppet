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

"""Tests for landmark-to-GNM-parameter solving.

The recovery tests drive the solver with synthetic landmarks sampled directly
from a GNM mesh posed with known parameters, which makes the ground truth exact
and isolates the solver from face-detection error.
"""

import functools

from absl.testing import absltest
from absl.testing import parameterized
from gnm.shape import gnm_numpy
import numpy as np
from scipy.spatial import transform as scipy_transform

from webcam_puppet import correspondence as correspondence_lib
from webcam_puppet import solver as solver_lib

_LEFT_EYE_BLOCK_START = 0
_LOWER_FACE_BLOCK_START = 200


@functools.cache
def _load_gnm() -> gnm_numpy.GNM:
  """Loads the GNM head model once for the whole test module."""
  return gnm_numpy.GNM.from_local(
      version=gnm_numpy.GNMMajorVersion.V3,
      variant=gnm_numpy.GNMVariant.HEAD,
  )


@functools.cache
def _synthetic_correspondence() -> correspondence_lib.Correspondence:
  """Builds a correspondence without invoking the face detector.

  Uses a strided sample of skin vertices as stand-in landmarks so these tests
  stay independent of MediaPipe and of the cached calibration asset.

  Returns:
    A correspondence over sampled skin vertices.
  """
  gnm = _load_gnm()
  head_joint = list(gnm.joint_names).index('head')

  skin = gnm.vertex_group_indices('skin')
  # Mirror build_correspondence: restrict to vertices rigidly skinned to the
  # head, so neck vertices cannot pose as rigid anchors.
  head_bound = skin[gnm.skinning_weights[head_joint][skin] >= 0.9]
  sampled = head_bound[:: max(len(head_bound) // 400, 1)]

  energy = correspondence_lib.expression_displacement_energy(gnm)[sampled]
  rigid = energy <= np.quantile(energy, 0.35)

  # With a synthetic correspondence the landmarks *are* the vertices, so the
  # neutral reference is the neutral mesh at those vertices. A real calibration
  # records where the landmarker actually places them instead.
  return correspondence_lib.Correspondence(
      landmark_indices=np.arange(len(sampled), dtype=np.int32),
      vertex_indices=sampled.astype(np.int32),
      rigid=rigid,
      reference_landmarks=gnm.template_vertex_positions[sampled].astype(
          np.float32
      ),
  )


def _model_points_to_landmarks(points: np.ndarray) -> np.ndarray:
  """Inverts `landmarks_to_model_axes` for a square image.

  Args:
    points: Points in GNM model space, (M, 3).

  Returns:
    Landmarks in MediaPipe's axis convention, (M, 3).
  """
  return np.stack([points[:, 0], -points[:, 1], -points[:, 2]], axis=-1)


class FitSimilarityTransformTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('identity', 0.0, 1.0),
      ('yaw_and_scale', 25.0, 1.4),
      ('negative_yaw_shrink', -40.0, 0.6),
  )
  def test_recovers_known_transform(self, yaw_degrees, scale):
    rng = np.random.default_rng(0)
    source = rng.normal(size=(40, 3))
    rotation = scipy_transform.Rotation.from_euler(
        'y', yaw_degrees, degrees=True
    ).as_matrix()
    translation = np.array([0.3, -0.2, 0.1])
    target = scale * source @ rotation.T + translation

    fitted_scale, fitted_rotation, fitted_translation = (
        solver_lib.fit_similarity_transform(source, target)
    )

    self.assertAlmostEqual(fitted_scale, scale, places=5)
    np.testing.assert_allclose(fitted_rotation, rotation, atol=1e-6)
    np.testing.assert_allclose(fitted_translation, translation, atol=1e-6)

  def test_does_not_return_a_reflection(self):
    rng = np.random.default_rng(1)
    source = rng.normal(size=(30, 3))
    # A mirrored target must still yield a proper rotation (determinant +1).
    target = source * np.array([1.0, 1.0, -1.0])

    _, rotation, _ = solver_lib.fit_similarity_transform(source, target)

    self.assertGreater(float(np.linalg.det(rotation)), 0.0)

  def test_rejects_mismatched_shapes(self):
    with self.assertRaisesRegex(ValueError, 'same shape'):
      solver_lib.fit_similarity_transform(
          np.zeros((5, 3)), np.zeros((6, 3))
      )

  def test_rejects_too_few_points(self):
    with self.assertRaisesRegex(ValueError, 'at least 3 points'):
      solver_lib.fit_similarity_transform(np.zeros((2, 3)), np.zeros((2, 3)))


class LandmarksToModelAxesTest(parameterized.TestCase):

  def test_flips_y_and_z_into_gnm_convention(self):
    landmarks = np.array([[0.5, 0.25, -0.1]], dtype=np.float32)

    converted = solver_lib.landmarks_to_model_axes(landmarks, 100, 100)

    np.testing.assert_allclose(converted, [[0.5, -0.25, 0.1]], atol=1e-6)

  def test_applies_aspect_ratio_to_x_and_z(self):
    landmarks = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)

    converted = solver_lib.landmarks_to_model_axes(landmarks, 200, 100)

    # x and z share MediaPipe's width normalization, y uses height.
    np.testing.assert_allclose(converted, [[2.0, -1.0, -2.0]], atol=1e-6)


class SelectExpressionComponentsTest(parameterized.TestCase):

  def test_blocks_cover_the_whole_basis_without_overlap(self):
    gnm = _load_gnm()

    blocks = solver_lib.expression_region_blocks(gnm)

    combined = np.concatenate(list(blocks.values()))
    np.testing.assert_array_equal(
        np.sort(combined), np.arange(gnm.expression_dim)
    )

  def test_expected_regions_are_discovered(self):
    gnm = _load_gnm()

    blocks = solver_lib.expression_region_blocks(gnm)

    self.assertContainsSubset(
        ['left_eye_region', 'right_eye_region', 'lower_face_region'],
        list(blocks),
    )

  def test_selection_spans_lower_face_not_just_leading_components(self):
    gnm = _load_gnm()

    selected = solver_lib.select_expression_components(gnm)

    # The regression this guards: taking the leading N components across the
    # whole basis would select eye components only, leaving the mouth rigid.
    self.assertTrue(np.any(selected >= _LOWER_FACE_BLOCK_START))
    self.assertTrue(np.any(selected < 100))

  def test_budget_limits_components_per_region(self):
    gnm = _load_gnm()

    selected = solver_lib.select_expression_components(
        gnm, {'lower_face_region': 5}
    )

    np.testing.assert_array_equal(
        selected,
        np.arange(_LOWER_FACE_BLOCK_START, _LOWER_FACE_BLOCK_START + 5),
    )

  def test_rejects_unknown_region(self):
    gnm = _load_gnm()

    with self.assertRaisesRegex(ValueError, 'Unknown expression region'):
      solver_lib.select_expression_components(gnm, {'nose_wiggle': 4})

  def test_rejects_empty_selection(self):
    gnm = _load_gnm()

    with self.assertRaisesRegex(ValueError, 'No expression components'):
      solver_lib.select_expression_components(gnm, {'lower_face_region': 0})


class LandmarkSolverTest(parameterized.TestCase):

  def _solve_for(
      self,
      expression: np.ndarray,
      yaw_degrees: float = 0.0,
      noise_metres: float = 0.0,
      seed: int = 0,
  ):
    """Poses the mesh, samples landmarks from it and solves them back."""
    gnm = _load_gnm()
    correspondence = _synthetic_correspondence()
    rng = np.random.default_rng(seed)

    rotations = np.zeros((gnm.num_joints, 3), dtype=np.float32)
    head_joint = list(gnm.joint_names).index('head')
    rotations[head_joint, 1] = np.deg2rad(yaw_degrees)

    posed = gnm(
        np.zeros(gnm.identity_dim, dtype=np.float32),
        expression,
        rotations,
        np.zeros(3, dtype=np.float32),
    )

    observed = posed[correspondence.vertex_indices]
    if noise_metres:
      observed = observed + rng.normal(0.0, noise_metres, observed.shape)

    landmarks = _model_points_to_landmarks(observed)
    solver = solver_lib.LandmarkSolver(
        gnm, correspondence, smoothing=0.0
    )
    return gnm, correspondence, posed, solver.solve(landmarks, 512, 512)

  def test_neutral_input_yields_near_zero_expression(self):
    gnm = _load_gnm()
    neutral = np.zeros(gnm.expression_dim, dtype=np.float32)

    _, _, _, parameters = self._solve_for(neutral)

    self.assertLess(float(np.abs(parameters.expression).max()), 1e-2)
    self.assertLess(parameters.landmark_rmse, 1e-4)

  @parameterized.named_parameters(
      ('no_yaw', 0.0),
      ('yaw_15', 15.0),
      ('yaw_negative_25', -25.0),
  )
  def test_recovers_head_yaw(self, yaw_degrees):
    gnm = _load_gnm()
    neutral = np.zeros(gnm.expression_dim, dtype=np.float32)

    _, _, _, parameters = self._solve_for(neutral, yaw_degrees=yaw_degrees)

    head_joint = list(gnm.joint_names).index('head')
    recovered = scipy_transform.Rotation.from_rotvec(
        parameters.rotations[head_joint]
    ).as_euler('xyz', degrees=True)
    self.assertAlmostEqual(float(recovered[1]), yaw_degrees, delta=1.0)

  def test_head_pose_lands_on_the_head_joint_only(self):
    gnm = _load_gnm()
    neutral = np.zeros(gnm.expression_dim, dtype=np.float32)

    _, _, _, parameters = self._solve_for(neutral, yaw_degrees=20.0)

    head_joint = list(gnm.joint_names).index('head')
    for joint in range(gnm.num_joints):
      if joint == head_joint:
        self.assertGreater(
            float(np.abs(parameters.rotations[joint]).max()), 0.1
        )
      else:
        np.testing.assert_allclose(
            parameters.rotations[joint], np.zeros(3), atol=1e-6
        )

  def test_recovers_most_of_a_lower_face_expression(self):
    gnm = _load_gnm()
    rng = np.random.default_rng(7)
    expression = np.zeros(gnm.expression_dim, dtype=np.float32)
    expression[_LOWER_FACE_BLOCK_START : _LOWER_FACE_BLOCK_START + 8] = (
        rng.normal(0.0, 2.0, 8)
    )

    _, correspondence, posed, parameters = self._solve_for(expression)

    reconstructed = gnm(
        parameters.identity,
        parameters.expression,
        parameters.rotations,
        parameters.translation,
    )
    tracked = correspondence.vertex_indices
    neutral_mesh = gnm.template_vertex_positions

    deformation = np.linalg.norm(
        posed[tracked] - neutral_mesh[tracked], axis=-1
    ).mean()
    error = np.linalg.norm(
        reconstructed[tracked] - posed[tracked], axis=-1
    ).mean()

    # Recovering most of the deformation is what matters; exact coefficient
    # recovery is not expected because the basis is not orthogonal at a sparse
    # vertex subset, so different coefficients explain the same geometry.
    self.assertLess(error, 0.4 * deformation)

  def test_degrades_gracefully_with_landmark_noise(self):
    gnm = _load_gnm()
    rng = np.random.default_rng(8)
    expression = np.zeros(gnm.expression_dim, dtype=np.float32)
    expression[_LOWER_FACE_BLOCK_START : _LOWER_FACE_BLOCK_START + 8] = (
        rng.normal(0.0, 2.0, 8)
    )

    _, _, _, clean = self._solve_for(expression, noise_metres=0.0, seed=3)
    _, _, _, noisy = self._solve_for(expression, noise_metres=0.002, seed=3)

    # 2mm of per-landmark noise is well beyond what the tracker produces. The
    # guard is against coefficients exploding into the basis's near-null
    # directions, not against a modest inflation: GNM's documented usable
    # coefficient range is roughly -3 to 3, so an order of magnitude past that
    # would indicate a genuine blow-up.
    self.assertLess(float(np.abs(noisy.expression).max()), 30.0)
    self.assertLess(
        float(np.abs(noisy.expression).max()),
        10.0 * float(np.abs(clean.expression).max()),
    )
    self.assertLess(noisy.landmark_rmse, 0.01)

  def test_tongue_and_pupil_components_are_left_unsolved(self):
    gnm = _load_gnm()
    rng = np.random.default_rng(9)
    expression = np.zeros(gnm.expression_dim, dtype=np.float32)
    expression[_LOWER_FACE_BLOCK_START : _LOWER_FACE_BLOCK_START + 8] = (
        rng.normal(0.0, 2.0, 8)
    )

    _, _, _, parameters = self._solve_for(expression)

    blocks = solver_lib.expression_region_blocks(gnm)
    for region in ('tongue', 'pupils'):
      np.testing.assert_allclose(
          parameters.expression[blocks[region]],
          np.zeros(len(blocks[region])),
          atol=0.0,
      )

  def test_smoothing_lags_a_step_change(self):
    gnm = _load_gnm()
    correspondence = _synthetic_correspondence()
    rng = np.random.default_rng(10)
    expression = np.zeros(gnm.expression_dim, dtype=np.float32)
    expression[_LOWER_FACE_BLOCK_START : _LOWER_FACE_BLOCK_START + 8] = (
        rng.normal(0.0, 2.0, 8)
    )

    neutral_mesh = gnm.template_vertex_positions
    posed = gnm(
        np.zeros(gnm.identity_dim, dtype=np.float32),
        expression,
        np.zeros((gnm.num_joints, 3), dtype=np.float32),
        np.zeros(3, dtype=np.float32),
    )
    neutral_landmarks = _model_points_to_landmarks(
        neutral_mesh[correspondence.vertex_indices]
    )
    posed_landmarks = _model_points_to_landmarks(
        posed[correspondence.vertex_indices]
    )

    solver = solver_lib.LandmarkSolver(
        gnm, correspondence, smoothing=0.8
    )
    solver.solve(neutral_landmarks, 512, 512)
    first_posed = solver.solve(posed_landmarks, 512, 512)

    unsmoothed = solver_lib.LandmarkSolver(
        gnm, correspondence, smoothing=0.0
    ).solve(posed_landmarks, 512, 512)

    self.assertLess(
        float(np.abs(first_posed.expression).max()),
        float(np.abs(unsmoothed.expression).max()),
    )

  def test_solve_identity_changes_the_neutral_mesh(self):
    gnm = _load_gnm()
    correspondence = _synthetic_correspondence()
    rng = np.random.default_rng(11)

    identity = np.zeros(gnm.identity_dim, dtype=np.float32)
    identity[:10] = rng.normal(0.0, 1.0, 10)
    target = gnm(
        identity,
        np.zeros(gnm.expression_dim, dtype=np.float32),
        np.zeros((gnm.num_joints, 3), dtype=np.float32),
        np.zeros(3, dtype=np.float32),
    )
    landmarks = _model_points_to_landmarks(
        target[correspondence.vertex_indices]
    )

    solver = solver_lib.LandmarkSolver(gnm, correspondence, smoothing=0.0)
    self.assertFalse(np.any(solver.identity))

    solved = solver.solve_identity(landmarks, 512, 512)

    self.assertTrue(np.any(solved))
    solver.reset()
    self.assertFalse(np.any(solver.identity))

  def test_rejects_out_of_range_smoothing(self):
    gnm = _load_gnm()
    correspondence = _synthetic_correspondence()

    with self.assertRaisesRegex(ValueError, 'smoothing'):
      solver_lib.LandmarkSolver(gnm, correspondence, smoothing=1.0)


class CorrespondenceTest(parameterized.TestCase):

  def test_save_load_roundtrip(self):
    correspondence = _synthetic_correspondence()
    path = self.create_tempdir().full_path + '/correspondence.npz'

    correspondence.save(path)
    loaded = correspondence_lib.Correspondence.load(path)

    np.testing.assert_array_equal(
        loaded.vertex_indices, correspondence.vertex_indices
    )
    np.testing.assert_array_equal(
        loaded.landmark_indices, correspondence.landmark_indices
    )
    np.testing.assert_array_equal(loaded.rigid, correspondence.rigid)

  def test_rejects_mismatched_lengths(self):
    with self.assertRaisesRegex(ValueError, 'same length'):
      correspondence_lib.Correspondence(
          landmark_indices=np.arange(3),
          vertex_indices=np.arange(4),
          rigid=np.ones(3, dtype=bool),
          reference_landmarks=np.zeros((3, 3)),
      )

  def test_rejects_no_rigid_anchors(self):
    with self.assertRaisesRegex(ValueError, 'rigid'):
      correspondence_lib.Correspondence(
          landmark_indices=np.arange(3),
          vertex_indices=np.arange(3),
          rigid=np.zeros(3, dtype=bool),
          reference_landmarks=np.zeros((3, 3)),
      )

  def test_rigid_anchors_move_less_than_the_rest(self):
    gnm = _load_gnm()
    correspondence = _synthetic_correspondence()

    energy = correspondence_lib.expression_displacement_energy(gnm)[
        correspondence.vertex_indices
    ]

    self.assertLess(
        float(energy[correspondence.rigid].mean()),
        float(energy[~correspondence.rigid].mean()),
    )

  def test_expression_energy_is_per_vertex_and_non_negative(self):
    gnm = _load_gnm()

    energy = correspondence_lib.expression_displacement_energy(gnm)

    self.assertEqual(energy.shape, (len(gnm.template_vertex_positions),))
    self.assertTrue(np.all(energy >= 0.0))


if __name__ == '__main__':
  absltest.main()
