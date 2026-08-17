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

"""Geometry shared by correspondence derivation and per-frame solving."""

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating]


def landmarks_to_model_axes(
    landmarks: FloatArray,
    image_width: int,
    image_height: int,
) -> FloatArray:
  """Converts MediaPipe normalized landmarks to GNM's axis convention.

  MediaPipe normalizes x by image width and y by image height, so the raw values
  are anisotropic whenever the frame is not square. Its z grows away from the
  camera and its y grows downward, both opposite to GNM.

  The result is only defined up to a global scale, which the similarity fit
  solves for; what matters is that the three axes share one scale.

  Args:
    landmarks: MediaPipe landmarks, (M, 3).
    image_width: Frame width in pixels.
    image_height: Frame height in pixels.

  Returns:
    Landmarks in GNM's Y-up, Z-towards-viewer convention, (M, 3).
  """
  aspect_ratio = image_width / image_height
  return np.stack(
      [
          landmarks[:, 0] * aspect_ratio,
          -landmarks[:, 1],
          # z shares x's normalization, so it takes the same aspect scaling.
          -landmarks[:, 2] * aspect_ratio,
      ],
      axis=-1,
  )


def fit_similarity_transform(
    source: FloatArray,
    target: FloatArray,
) -> tuple[float, FloatArray, FloatArray]:
  """Fits the similarity transform taking source onto target.

  Solves for the scale, rotation and translation minimizing
  `|| scale * rotation @ source + translation - target ||^2` in closed form
  (Umeyama's method).

  Args:
    source: Source points, (N, 3).
    target: Target points, (N, 3).

  Returns:
    A tuple of scale, rotation matrix (3, 3), and translation (3,).

  Raises:
    ValueError: If fewer than three points are given, or shapes disagree.
  """
  if source.shape != target.shape:
    raise ValueError(
        f'source and target must have the same shape, got {source.shape} and'
        f' {target.shape}.'
    )
  if len(source) < 3:
    raise ValueError(f'Need at least 3 points to fit, got {len(source)}.')

  source_centroid = source.mean(axis=0)
  target_centroid = target.mean(axis=0)
  source_centered = source - source_centroid
  target_centered = target - target_centroid

  covariance = target_centered.T @ source_centered / len(source)
  left, singular_values, right_transposed = np.linalg.svd(covariance)

  # Guard against the SVD returning a reflection instead of a rotation.
  correction = np.ones(3)
  if np.linalg.det(left @ right_transposed) < 0:
    correction[-1] = -1.0

  rotation = left @ np.diag(correction) @ right_transposed

  source_variance = np.sum(np.square(source_centered)) / len(source)
  scale = float(
      np.sum(singular_values * correction) / max(source_variance, 1e-12)
  )

  translation = target_centroid - scale * rotation @ source_centroid
  return scale, rotation, translation


def apply_similarity_transform(
    points: FloatArray,
    scale: float,
    rotation: FloatArray,
    translation: FloatArray,
) -> FloatArray:
  """Applies a similarity transform to points.

  Args:
    points: Points to transform, (N, 3).
    scale: Uniform scale factor.
    rotation: Rotation matrix, (3, 3).
    translation: Translation vector, (3,).

  Returns:
    The transformed points, (N, 3).
  """
  return scale * points @ rotation.T + translation
