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

"""MediaPipe face landmarker wrapper."""

import dataclasses
import urllib.request

from etils import epath
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating]

MODEL_URL = (
    'https://storage.googleapis.com/mediapipe-models/face_landmarker/'
    'face_landmarker/float16/1/face_landmarker.task'
)

_DEFAULT_MODEL_PATH = (
    epath.resource_path('webcam_puppet') / 'assets' / 'face_landmarker.task'
)


def ensure_model(path: epath.PathLike | None = None) -> epath.Path:
  """Returns the landmarker model path, downloading it if it is missing.

  Args:
    path: Where the .task model should live. Defaults to the packaged assets
      directory.

  Returns:
    The path to the model file.
  """
  path = epath.Path(path) if path is not None else _DEFAULT_MODEL_PATH
  if path.exists():
    return path

  path.parent.mkdir(parents=True, exist_ok=True)
  print(f'Downloading face landmarker model to {path} ...')
  with urllib.request.urlopen(MODEL_URL) as response:
    payload = response.read()
  path.write_bytes(payload)
  return path


@dataclasses.dataclass(frozen=True)
class FaceObservation:
  """A single detected face.

  Attributes:
    landmarks: Landmark positions, (M, 3). x and y are normalized to [0, 1] of
      the image width and height; z is a relative depth in roughly the same
      units as x, and is not metric.
    blendshapes: The 52 ARKit-style blendshape scores in [0, 1], or None if the
      landmarker was not asked for them.
  """

  landmarks: FloatArray
  blendshapes: FloatArray | None = None


class FaceTracker:
  """Detects face landmarks in images or a video stream."""

  def __init__(
      self,
      model_path: epath.PathLike | None = None,
      video_mode: bool = False,
      output_blendshapes: bool = False,
  ):
    """Builds the tracker.

    Args:
      model_path: Path to the landmarker .task model; downloaded when absent.
      video_mode: If True, use MediaPipe's VIDEO running mode, which applies
        temporal tracking across frames. Requires monotonically increasing
        timestamps in `detect`.
      output_blendshapes: If True, also return the 52 blendshape scores.
    """
    self._video_mode = video_mode
    resolved_model_path = ensure_model(model_path)

    self._landmarker = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(resolved_model_path)
            ),
            running_mode=(
                vision.RunningMode.VIDEO
                if video_mode
                else vision.RunningMode.IMAGE
            ),
            num_faces=1,
            output_face_blendshapes=output_blendshapes,
        )
    )

  def detect(
      self,
      image: npt.NDArray[np.uint8],
      timestamp_ms: int | None = None,
  ) -> FaceObservation | None:
    """Detects the most prominent face in an RGB image.

    Args:
      image: RGB uint8 image, (H, W, 3).
      timestamp_ms: Frame timestamp, required in video mode and ignored
        otherwise. Must strictly increase between calls.

    Returns:
      The detected face, or None if no face was found.

    Raises:
      ValueError: If video mode is active but no timestamp was supplied.
    """
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image)
    )

    if self._video_mode:
      if timestamp_ms is None:
        raise ValueError('timestamp_ms is required in video mode.')
      result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
    else:
      result = self._landmarker.detect(mp_image)

    if not result.face_landmarks:
      return None

    landmarks = np.array(
        [[p.x, p.y, p.z] for p in result.face_landmarks[0]], dtype=np.float32
    )

    blendshapes = None
    if result.face_blendshapes:
      blendshapes = np.array(
          [c.score for c in result.face_blendshapes[0]], dtype=np.float32
      )

    return FaceObservation(landmarks=landmarks, blendshapes=blendshapes)

  def landmarks_only(
      self, image: npt.NDArray[np.uint8]
  ) -> FloatArray | None:
    """Detects a face and returns just its landmarks, for calibration."""
    observation = self.detect(image)
    return None if observation is None else observation.landmarks

  def close(self) -> None:
    """Releases the underlying landmarker."""
    self._landmarker.close()

  def __enter__(self) -> 'FaceTracker':
    return self

  def __exit__(self, *exc_info) -> None:
    self.close()
