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

"""Derives a MediaPipe-landmark to GNM-vertex correspondence.

MediaPipe's face mesh and GNM use unrelated topologies, so driving GNM from
MediaPipe needs a mapping between them. Rather than hand-transcribing landmark
indices -- which is error prone and hard to verify -- this module derives the
mapping by construction:

  1. Render the GNM head in several known poses with a known camera.
  2. Run the MediaPipe face landmarker on each render.
  3. Back-project each detected landmark onto the nearest visible GNM skin
     vertex, which is exact up to the projected vertex spacing because the
     camera is known.
  4. Merge the per-pose matches, giving every landmark a vertex of its own.

Steps 1 and 4 are why this is *multi-shot*. A neutral render cannot separate the
lips: with the mouth closed, MediaPipe's upper- and lower-lip landmarks project
onto the same GNM vertex, and the one-landmark-per-vertex rule then discards
half of every pair -- which cost 12 of the 40 lip landmarks and left 8 of 9
upper/lower pairs sharing a single vertex. Rendering the head in several mouth
poses as well, and taking each landmark from whichever pose placed it best,
recovers all 40 and all 9 pairs.

Step 3 tests visibility rather than facing direction. A vertex whose normal
points at the camera is not necessarily visible: the far wall of an open mouth
faces the camera too, and matching a lip landmark to it puts the correspondence
inside the head. Candidates are therefore rejected when a nearer candidate
covers the same pixel.

The rigid subset used later for pose estimation is derived from the model too:
vertices whose expression-basis displacement energy is lowest barely move with
expression, which makes them the natural skull-fixed anchors.
"""

from collections.abc import Sequence
import dataclasses

from absl import logging
from etils import epath
from gnm.shape import gnm_numpy
import numpy as np
import numpy.typing as npt
import scipy.ndimage

from webcam_puppet import geometry
from webcam_puppet import renderer as renderer_lib

FloatArray = npt.NDArray[np.floating]
IntArray = npt.NDArray[np.integer]

# Resolution used for calibration only. Larger than the live render, since a
# finer projected vertex spacing makes the back-projection more precise.
_CALIBRATION_IMAGE_SIZE = (768, 768)

# The landmarker is trained on photographs, so calibration renders use a
# skin-like albedo on white rather than the app's stylized dark palette.
_CALIBRATION_BASE_COLOR = (0.80, 0.66, 0.58)
_CALIBRATION_BACKGROUND = (1.0, 1.0, 1.0)

# Fraction of corresponded landmarks kept as skull-fixed pose anchors.
_RIGID_QUANTILE = 0.35

# Mouth expression components used for calibration poses, beyond the neutral
# render. Each contributes two poses, one per sign. Two components is enough to
# give every lip landmark its own vertex; a third measured no better.
_MOUTH_COMPONENT_COUNT = 2

# Pose amplitude, as an expression coefficient. Held well inside the +/-3 the
# live solver allows, so these are poses the model genuinely reaches rather
# than extrapolations -- the landmarker has to believe the render is a face.
_MOUTH_POSE_AMPLITUDE = 1.5

# Expression components whose name carries this prefix drive the mouth.
_MOUTH_REGION_PREFIX = 'lower_face_region'

# Skin within this distance of a tooth is lip. Used only to shortlist which
# expression components move the mouth most, so the poses stay model-derived
# rather than hand-picked vertex indices.
_LIP_BAND_DISTANCE = 0.004

# A candidate is occluded when it lies more than this far behind the nearest
# candidate covering its pixel. Loose enough to keep a whole quad's worth of
# depth variation on a curved surface, tight enough to reject the far side of
# the lips.
_OCCLUSION_TOLERANCE = 0.003
_OCCLUSION_RADIUS_PIXELS = 2

_CACHE_VERSION = 2


class CalibrationError(Exception):
  """Raised when correspondence cannot be derived from a calibration render."""


@dataclasses.dataclass(frozen=True)
class Correspondence:
  """A mapping from MediaPipe landmarks to GNM vertices.

  Attributes:
    landmark_indices: Indices into MediaPipe's landmark array, (M,).
    vertex_indices: The GNM vertex matched to each landmark, (M,).
    rigid: Boolean mask selecting landmarks that barely move with expression,
      used to estimate head pose, (M,).
    reference_landmarks: Where the landmarker places these landmarks on a
      *neutral* face, expressed in GNM's metric model space, (M, 3).

      This is what makes tracking differential. The landmarker reports its own
      idea of face shape -- and a z that is not metric at all -- so its
      landmarks do not coincide with the GNM vertices they were matched to,
      even for the very render they were detected on. Fitting their absolute
      positions
      therefore asks GNM to deform into MediaPipe's face, which shows up as
      several millimetres of invented expression on a resting face. Fitting the
      *displacement* from this neutral reference instead cancels that constant
      bias, so a resting face yields zero expression by construction.
  """

  landmark_indices: IntArray
  vertex_indices: IntArray
  rigid: npt.NDArray[np.bool_]
  reference_landmarks: FloatArray

  def __post_init__(self):
    if not (
        len(self.landmark_indices)
        == len(self.vertex_indices)
        == len(self.rigid)
        == len(self.reference_landmarks)
    ):
      raise ValueError(
          'landmark_indices, vertex_indices, rigid and reference_landmarks must'
          f' be the same length, got {len(self.landmark_indices)},'
          f' {len(self.vertex_indices)}, {len(self.rigid)} and'
          f' {len(self.reference_landmarks)}.'
      )
    if self.reference_landmarks.ndim != 2 or (
        self.reference_landmarks.shape[1] != 3
    ):
      raise ValueError(
          'reference_landmarks must have shape (M, 3), got'
          f' {self.reference_landmarks.shape}.'
      )
    if not np.any(self.rigid):
      raise ValueError('At least one landmark must be marked rigid.')

  def save(self, path: epath.PathLike) -> None:
    """Writes the correspondence to an .npz file."""
    path = epath.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f:
      np.savez(
          f,
          version=np.asarray(_CACHE_VERSION),
          landmark_indices=self.landmark_indices,
          vertex_indices=self.vertex_indices,
          rigid=self.rigid,
          reference_landmarks=self.reference_landmarks,
      )

  @classmethod
  def load(cls, path: epath.PathLike) -> 'Correspondence':
    """Reads a correspondence written by `save`."""
    with epath.Path(path).open('rb') as f:
      data = dict(np.load(f))
    version = int(data.get('version', 0))
    if version != _CACHE_VERSION:
      raise CalibrationError(
          f'Correspondence cache version {version} does not match expected'
          f' {_CACHE_VERSION}; delete the cache to rebuild it.'
      )
    return cls(
        landmark_indices=data['landmark_indices'],
        vertex_indices=data['vertex_indices'],
        rigid=data['rigid'].astype(bool),
        reference_landmarks=data['reference_landmarks'],
    )


def expression_displacement_energy(gnm: gnm_numpy.GNM) -> FloatArray:
  """Measures how much each vertex moves across the expression basis.

  Args:
    gnm: The GNM model.

  Returns:
    Root-mean-square displacement magnitude per vertex, (V,).
  """
  # expression_basis is (E, V, 3); reduce over components and coordinates.
  squared = np.sum(np.square(gnm.expression_basis), axis=(0, 2))
  return np.sqrt(squared / gnm.expression_basis.shape[0])


def _lip_band_indices(gnm: gnm_numpy.GNM) -> tuple[IntArray, IntArray]:
  """Finds the skin vertices lining the upper and lower jaw.

  The model names its teeth groups but not its lips, so the lips are located
  by proximity: the skin that sheathes each dental arch is the band that parts
  when the mouth opens.

  Args:
    gnm: The GNM model.

  Returns:
    A tuple of the upper and lower band vertex indices.
  """
  vertices = np.asarray(gnm.template_vertex_positions)
  skin = np.asarray(gnm.vertex_group_indices('skin'))

  bands = []
  for group in ('upper_teeth_and_gums', 'lower_teeth_and_gums'):
    teeth = vertices[np.asarray(gnm.vertex_group_indices(group))]
    # (skin, teeth) distances: a few thousand by 1440 is small enough to take
    # densely, and this runs once per calibration.
    nearest = np.min(
        np.linalg.norm(vertices[skin][:, None, :] - teeth[None, :, :], axis=-1),
        axis=1,
    )
    bands.append(skin[nearest <= _LIP_BAND_DISTANCE])
  return bands[0], bands[1]


def mouth_calibration_poses(
    gnm: gnm_numpy.GNM,
    count: int = _MOUTH_COMPONENT_COUNT,
    amplitude: float = _MOUTH_POSE_AMPLITUDE,
) -> list[FloatArray]:
  """Builds calibration poses that move the mouth away from its closed rest.

  Shortlists the mouth-region expression components by how far they drive the
  two lip bands apart, then emits each at both signs.

  Both signs, rather than the one that opens the mouth, because there is no
  reliable cheap test for which sign that is: the lip bands include skin well
  away from the lip line, so a component that only purses or drops the chin can
  outrank one that genuinely parts the lips. Rendering both costs one detection
  and removes the guess -- a pose that closes an already-closed mouth simply
  wins no assignments in the merge. That leaves the shortlist only needing to
  surface components that move the mouth a lot, which it does reliably.

  Args:
    gnm: The GNM model.
    count: How many components to use; each yields two poses.
    amplitude: Expression coefficient magnitude for each pose.

  Returns:
    Posed vertex arrays, each (V, 3).
  """
  upper, lower = _lip_band_indices(gnm)
  basis = np.asarray(gnm.expression_basis)
  mouth = np.asarray([
      index
      for index, name in enumerate(gnm.expression_names)
      if str(name).startswith(_MOUTH_REGION_PREFIX)
  ])

  # How far one unit of each component separates the bands vertically.
  parting = (
      basis[mouth][:, lower, 1].mean(axis=1)
      - basis[mouth][:, upper, 1].mean(axis=1)
  )
  leading = mouth[np.argsort(-np.abs(parting))[:count]]

  neutral = np.asarray(gnm.template_vertex_positions)
  return [
      neutral + sign * amplitude * basis[component]
      for component in leading
      for sign in (1.0, -1.0)
  ]


def _visible_candidates(
    gnm: gnm_numpy.GNM,
    vertices: FloatArray,
    camera: renderer_lib.Camera,
    eligible: npt.NDArray[np.bool_],
) -> tuple[IntArray, FloatArray]:
  """Selects the vertices a landmark may legitimately be matched to.

  Args:
    gnm: The GNM model, for its triangles.
    vertices: Posed vertex positions, (V, 3).
    camera: The calibration camera.
    eligible: Per-vertex mask of pose-independent eligibility.

  Returns:
    A tuple of the surviving vertex indices and their projected pixels.
  """
  normals = renderer_lib.compute_vertex_normals(vertices, gnm.triangles)
  camera_position = camera.target + np.array(
      [0.0, 0.0, camera.distance], dtype=np.float32
  )
  facing = np.sum(normals * (camera_position - vertices), axis=-1) > 0.0

  indices = np.flatnonzero(eligible & facing)
  pixels, depth = camera.project(vertices[indices])

  # Facing the camera is not the same as being seen. Splat the candidates into
  # a depth buffer, spread each pixel's nearest depth over its neighbourhood so
  # a landmark sitting between two projected vertices is still compared against
  # the surface in front of it, then drop whatever sits behind that surface.
  height, width = camera.image_size
  columns = np.clip(np.round(pixels[:, 0]).astype(np.int32), 0, width - 1)
  rows = np.clip(np.round(pixels[:, 1]).astype(np.int32), 0, height - 1)
  buffer = np.full(height * width, np.inf, dtype=np.float64)
  np.minimum.at(buffer, rows * width + columns, depth)
  nearest = scipy.ndimage.minimum_filter(
      buffer.reshape(height, width), size=2 * _OCCLUSION_RADIUS_PIXELS + 1
  )
  visible = depth - nearest[rows, columns] <= _OCCLUSION_TOLERANCE

  return indices[visible], pixels[visible]


def build_correspondence(
    gnm: gnm_numpy.GNM,
    detect_landmarks,
    candidate_vertex_groups: Sequence[str] = ('skin',),
    image_size: tuple[int, int] = _CALIBRATION_IMAGE_SIZE,
    max_pixel_distance: float = 8.0,
    head_joint_name: str = 'head',
    min_head_skinning_weight: float = 0.9,
    mouth_component_count: int = _MOUTH_COMPONENT_COUNT,
) -> Correspondence:
  """Derives the landmark-to-vertex correspondence from calibration renders.

  Args:
    gnm: The GNM model to calibrate against.
    detect_landmarks: Callable taking an RGB uint8 image and returning
      normalized landmarks as an (M, 3) array with x and y in [0, 1], or None if
      no face was found.
    candidate_vertex_groups: Vertex groups eligible for matching. Restricting to
      skin keeps landmarks off the eyeballs, teeth and tongue, which sit behind
      the skin surface and would otherwise capture nearby landmarks.
    image_size: Calibration render resolution as (height, width).
    max_pixel_distance: Landmarks farther than this from any candidate vertex
      are dropped as unmatched.
    head_joint_name: Joint that head pose is estimated for.
    min_head_skinning_weight: Vertices must be at least this strongly skinned to
      the head joint to be matched. Without this, neck vertices qualify -- they
      have no expression energy, so they look maximally rigid, yet they stay put
      when the head joint rotates. Including them makes the similarity fit
      average moving and stationary points and systematically under-estimate
      head rotation.
    mouth_component_count: How many mouth expression components to build
      calibration poses from, each contributing one render per sign. Zero
      reproduces the single-shot behaviour, which loses most of the lip
      correspondences to vertex collisions.

  Returns:
    The derived correspondence.

  Raises:
    CalibrationError: If no face is detected or too few landmarks match.
    ValueError: If the head joint is not present in the model.
  """
  if head_joint_name not in list(gnm.joint_names):
    raise ValueError(
        f'Joint {head_joint_name!r} not in model joints'
        f' {list(gnm.joint_names)}.'
    )
  head_joint = list(gnm.joint_names).index(head_joint_name)
  neutral_vertices = gnm.template_vertex_positions
  camera = renderer_lib.Camera.fit_to_mesh(neutral_vertices, image_size)
  renderer = renderer_lib.MeshRenderer(
      gnm.triangles,
      camera,
      background=_CALIBRATION_BACKGROUND,
      base_color=_CALIBRATION_BASE_COLOR,
  )
  # Only vertices in the allowed groups can be matched: a landmark sits on the
  # visible surface, so matching it to an interior vertex would attach the
  # tracker inside the head.
  eligible = np.zeros(len(neutral_vertices), dtype=bool)
  for group in candidate_vertex_groups:
    eligible[gnm.vertex_group_indices(group)] = True

  # Only accept vertices that rigidly follow the head joint, so that pose
  # estimation sees points which actually move with the head.
  eligible &= gnm.skinning_weights[head_joint] >= min_head_skinning_weight

  height, width = image_size
  poses = [neutral_vertices] + mouth_calibration_poses(
      gnm, mouth_component_count
  )

  # Every landmark's best match in every pose, as (pose, distance, landmark,
  # vertex). Pose index leads, and the merge below sorts on it first.
  neutral_landmarks = None
  proposals: list[tuple[int, float, int, int]] = []
  for index, vertices in enumerate(poses):
    landmarks = detect_landmarks(renderer.render(vertices))
    if landmarks is None:
      if index == 0:
        raise CalibrationError(
            'The face landmarker found no face in the neutral calibration'
            ' render. This is usually a renderer or camera framing problem'
            ' rather than a model one.'
        )
      # A parting pose the detector refuses is a pose worth skipping, not a
      # failure: the neutral render already carries most of the face.
      logging.warning(
          'No face detected in calibration pose %d; skipping.', index
      )
      continue
    if index == 0:
      neutral_landmarks = landmarks

    candidate_indices, candidate_pixels = _visible_candidates(
        gnm, vertices, camera, eligible
    )
    if candidate_indices.size == 0:
      raise CalibrationError(
          'No visible candidate vertices in groups'
          f' {tuple(candidate_vertex_groups)}.'
      )

    target_pixels = np.stack(
        [landmarks[:, 0] * width, landmarks[:, 1] * height], axis=-1
    )
    # Nearest projected candidate per landmark. 478 x ~7k distances is small
    # enough to evaluate densely, and this runs once per calibration.
    deltas = target_pixels[:, None, :] - candidate_pixels[None, :, :]
    distances = np.linalg.norm(deltas, axis=-1)
    nearest = np.argmin(distances, axis=1)
    nearest_distance = distances[np.arange(len(target_pixels)), nearest]

    for landmark in np.flatnonzero(nearest_distance <= max_pixel_distance):
      proposals.append((
          index,
          float(nearest_distance[landmark]),
          int(landmark),
          int(candidate_indices[nearest[landmark]]),
      ))

  # Merge: neutral pose first, then by distance within a pose, and a landmark
  # or vertex is spent once it is used. Keeping vertices distinct is what
  # separates the lips -- both members of an upper/lower pair match the same
  # vertex on a closed mouth, so the loser falls through to a pose that parted
  # them. Greedy rather than optimal, which costs a fraction of a pixel and
  # stays readable.
  #
  # The neutral pose leads rather than the closest proposal overall, and that
  # ordering is load-bearing. A deformed render slides the surface under the
  # landmarks, so a posed match can be *nearer* in pixels while naming the
  # wrong vertex -- and the vertex is what selects the expression-basis rows
  # the live solver fits through. Letting distance rank across poses cost a
  # quarter of the recovered mouth opening when measured. Posed matches are
  # therefore a fallback for landmarks the neutral render could not place, not
  # a competitor to it.
  taken_landmark: set[int] = set()
  taken_vertex: set[int] = set()
  merged: list[tuple[int, int]] = []
  for _, _, landmark, vertex in sorted(proposals):
    if landmark in taken_landmark or vertex in taken_vertex:
      continue
    taken_landmark.add(landmark)
    taken_vertex.add(vertex)
    merged.append((landmark, vertex))

  if len(merged) < 3:
    raise CalibrationError(
        f'Only {len(merged)} landmarks matched within {max_pixel_distance}px;'
        f' expected most of {len(neutral_landmarks)}. Try raising'
        ' max_pixel_distance.'
    )

  merged.sort()
  landmark_indices = np.asarray([m[0] for m in merged], dtype=np.int32)
  vertex_indices = np.asarray([m[1] for m in merged], dtype=np.int32)

  # Skull-fixed anchors: the corresponded vertices that move least across the
  # expression basis. Deriving these from the basis avoids hand-picking indices.
  energy = expression_displacement_energy(gnm)[vertex_indices]
  threshold = np.quantile(energy, _RIGID_QUANTILE)
  rigid = energy <= threshold

  # Record the neutral landmark cloud in metric model space. The landmarker's
  # own units are arbitrary, so fit the similarity transform onto the matched
  # GNM vertices using the rigid anchors and carry the whole cloud through it.
  #
  # Always the *neutral* detection, whichever pose supplied a landmark's
  # vertex: this is the resting cloud the live solver measures displacement
  # from, so taking it from a parted-lip pose would bake that pose in as the
  # subject's neutral face.
  observed = geometry.landmarks_to_model_axes(
      neutral_landmarks, width, height
  )[landmark_indices]
  scale, rotation, translation = geometry.fit_similarity_transform(
      observed[rigid], neutral_vertices[vertex_indices][rigid]
  )
  reference_landmarks = geometry.apply_similarity_transform(
      observed, scale, rotation, translation
  ).astype(np.float32)

  return Correspondence(
      landmark_indices=landmark_indices,
      vertex_indices=vertex_indices,
      rigid=rigid,
      reference_landmarks=reference_landmarks,
  )


def load_or_build_correspondence(
    gnm: gnm_numpy.GNM,
    detect_landmarks,
    cache_path: epath.PathLike,
    rebuild: bool = False,
    **build_kwargs,
) -> Correspondence:
  """Loads a cached correspondence, deriving and caching it when absent.

  Args:
    gnm: The GNM model to calibrate against.
    detect_landmarks: See `build_correspondence`.
    cache_path: Where the correspondence .npz lives.
    rebuild: If True, ignore any existing cache and derive it again.
    **build_kwargs: Forwarded to `build_correspondence`.

  Returns:
    The correspondence.
  """
  cache_path = epath.Path(cache_path)
  if not rebuild and cache_path.exists():
    return Correspondence.load(cache_path)

  correspondence = build_correspondence(gnm, detect_landmarks, **build_kwargs)
  correspondence.save(cache_path)
  return correspondence
