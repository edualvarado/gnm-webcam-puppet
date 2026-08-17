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

"""Opening the webcam a human would have picked.

Three separate things go wrong when selecting a camera by index, and all of them
were observed on the Windows laptop this was developed against:

  1. `isOpened()` is not a readiness check. A device can open and then fail
     every single read.
  2. Cameras need to warm up. The first read or two after opening can fail on
     hardware that works fine a fraction of a second later, so a single failed
     read does not mean the device is unusable.
  3. Delivering frames is not the same as showing a picture. The Windows Hello
     infrared sensor typically enumerates as index 0, ahead of the real webcam,
     and hands back frames that are nearly black, pure greyscale and completely
     static. Anything that merely checks "did a frame arrive" selects it and the
     app appears to have no webcam.

Auto-selection therefore scores candidates on how alive they look -- colour
content plus frame-to-frame change -- and takes the best. On the machine above
the infrared device scores 0.1 and the real webcam scores ~25, so the two are
never confused. An explicit index bypasses scoring entirely.
"""

from collections.abc import Callable, Sequence
import time

from absl import logging
import cv2
import numpy as np
import numpy.typing as npt

# How long to keep retrying reads while a freshly opened camera warms up.
_WARMUP_SECONDS = 2.0
_WARMUP_POLL_SECONDS = 0.05

# Consecutive failed reads tolerated mid-session before giving up. Dropped
# frames happen; one is not a reason to end a live session.
_MAX_CONSECUTIVE_FAILURES = 30

# Indices probed when searching for a working camera.
_PROBE_LIMIT = 5

# Frames sampled per candidate when scoring how alive a device looks. Needs to
# be at least 2 for frame-to-frame change to be measurable.
_SCORING_FRAMES = 4

# Below this liveness score a device is almost certainly not a real webcam:
# measured 0.11 for a Windows Hello infrared sensor versus ~25 for the webcam
# beside it.
_BLANK_SCORE_THRESHOLD = 1.0

# Pause after releasing a device during probing. Windows keeps the capture graph
# busy briefly, which makes an immediately following open of another index fail.
_RELEASE_SETTLE_SECONDS = 0.3

CaptureFactory = Callable[[int], cv2.VideoCapture]


class NoWorkingCameraError(Exception):
  """Raised when no camera can deliver frames."""


def _default_capture_factory(index: int) -> cv2.VideoCapture:
  """Opens a capture device, preferring DirectShow on Windows."""
  # DirectShow avoids the Media Foundation backend's long stalls and spurious
  # read errors on Windows. On other platforms the flag is simply ignored.
  return cv2.VideoCapture(index, cv2.CAP_DSHOW)


def _read_with_warmup(
    capture: cv2.VideoCapture,
    timeout_seconds: float = _WARMUP_SECONDS,
) -> npt.NDArray[np.uint8] | None:
  """Reads one frame, retrying while the device warms up.

  Args:
    capture: An opened capture device.
    timeout_seconds: How long to keep retrying.

  Returns:
    The first frame read, or None if none arrived before the timeout.
  """
  deadline = time.monotonic() + timeout_seconds
  while time.monotonic() < deadline:
    ok, frame = capture.read()
    if ok and frame is not None:
      return frame
    time.sleep(_WARMUP_POLL_SECONDS)
  return None


def liveness_score(frames: Sequence[npt.NDArray[np.uint8]]) -> float:
  """Scores how much a sample of frames looks like a real camera feed.

  Combines two signals that both distinguish a webcam from an infrared or
  virtual device: colour content, since infrared sensors emit greyscale where
  the three channels are near-identical, and frame-to-frame change, since a
  sensor with its emitter off returns a frozen image.

  Args:
    frames: Consecutive BGR frames from one device.

  Returns:
    A non-negative score. Higher looks more like a live colour camera; an
    infrared sensor scores near zero.
  """
  if not frames:
    return 0.0

  sample = frames[-1].astype(np.int16)
  chroma = (
      np.abs(sample[:, :, 0] - sample[:, :, 1]).mean()
      + np.abs(sample[:, :, 1] - sample[:, :, 2]).mean()
  ) / 2.0

  motion = 0.0
  if len(frames) > 1:
    differences = [
        np.abs(
            frames[index].astype(np.int16) - frames[index - 1].astype(np.int16)
        ).mean()
        for index in range(1, len(frames))
    ]
    motion = float(np.mean(differences))

  return float(chroma) + motion


def _sample_frames(
    capture: cv2.VideoCapture,
    count: int = _SCORING_FRAMES,
    warmup_seconds: float = _WARMUP_SECONDS,
) -> list[npt.NDArray[np.uint8]]:
  """Reads up to `count` consecutive frames, allowing for warm-up.

  Args:
    capture: An opened capture device.
    count: How many frames to try to collect.
    warmup_seconds: How long to wait for the first frame.

  Returns:
    The frames collected, which is empty if the device never delivered one.
  """
  first = _read_with_warmup(capture, warmup_seconds)
  if first is None:
    return []

  frames = [first]
  while len(frames) < count:
    ok, frame = capture.read()
    if not ok or frame is None:
      break
    frames.append(frame)
  return frames


class FrameSource:
  """A camera that has been verified to deliver frames."""

  def __init__(self, capture: cv2.VideoCapture, index: int):
    """Wraps an opened, verified capture device.

    Args:
      capture: The opened capture device.
      index: The device index it was opened from, for logging.
    """
    self._capture = capture
    self._index = index
    self._consecutive_failures = 0

  @property
  def index(self) -> int:
    """The device index this source reads from."""
    return self._index

  def read(self) -> npt.NDArray[np.uint8] | None:
    """Reads the next frame.

    Transient failures are retried; None is returned only once failures persist,
    which indicates the device has genuinely gone away.

    Returns:
      A BGR frame, or None if the camera has stopped delivering.
    """
    ok, frame = self._capture.read()
    if ok and frame is not None:
      self._consecutive_failures = 0
      return frame

    self._consecutive_failures += 1
    if self._consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
      logging.error(
          'Camera %d failed %d consecutive reads; giving up.',
          self._index,
          self._consecutive_failures,
      )
      return None

    # Brief pause so a hiccup does not spin the loop at full speed.
    time.sleep(_WARMUP_POLL_SECONDS)
    return self.read()

  def release(self) -> None:
    """Releases the underlying device."""
    self._capture.release()

  def __enter__(self) -> 'FrameSource':
    return self

  def __exit__(self, *exc_info) -> None:
    self.release()


def probe_cameras(
    limit: int = _PROBE_LIMIT,
    capture_factory: CaptureFactory = _default_capture_factory,
    warmup_seconds: float = _WARMUP_SECONDS,
    settle_seconds: float = _RELEASE_SETTLE_SECONDS,
) -> list[int]:
  """Finds device indices that actually deliver a frame.

  Args:
    limit: Number of indices to probe, starting from 0.
    capture_factory: Opens a capture device for an index. Injectable for tests.
    warmup_seconds: How long to wait for each device to produce a frame.
    settle_seconds: Pause after releasing each device.

  Returns:
    The working indices, in probe order.
  """
  return sorted(
      score_cameras(limit, capture_factory, warmup_seconds, settle_seconds)
  )


def score_cameras(
    limit: int = _PROBE_LIMIT,
    capture_factory: CaptureFactory = _default_capture_factory,
    warmup_seconds: float = _WARMUP_SECONDS,
    settle_seconds: float = _RELEASE_SETTLE_SECONDS,
) -> dict[int, float]:
  """Scores every device that delivers frames by how alive it looks.

  Args:
    limit: Number of indices to probe, starting from 0.
    capture_factory: Opens a capture device for an index. Injectable for tests.
    warmup_seconds: How long to wait for each device to produce a frame.
    settle_seconds: Pause after releasing each device, so the platform can tear
      the capture graph down before the next open.

  Returns:
    A mapping from working device index to its liveness score. Devices that
    deliver nothing are absent.
  """
  scores: dict[int, float] = {}
  for index in range(limit):
    capture = capture_factory(index)
    try:
      if not capture.isOpened():
        continue
      frames = _sample_frames(capture, _SCORING_FRAMES, warmup_seconds)
      if frames:
        scores[index] = liveness_score(frames)
    finally:
      capture.release()
      time.sleep(settle_seconds)
  return scores


def open_camera(
    index: int | None = None,
    capture_factory: CaptureFactory = _default_capture_factory,
    probe_limit: int = _PROBE_LIMIT,
    warmup_seconds: float = _WARMUP_SECONDS,
    settle_seconds: float = _RELEASE_SETTLE_SECONDS,
) -> FrameSource:
  """Opens a camera, verifying it delivers frames before returning.

  Args:
    index: Device index to open, or None to auto-select the first working one.
    capture_factory: Opens a capture device for an index. Injectable for tests.
    probe_limit: Number of indices to probe when auto-selecting or when
      reporting alternatives in an error.
    warmup_seconds: How long to wait for a device to produce its first frame.
    settle_seconds: Pause after releasing a rejected device.

  Returns:
    A verified frame source.

  Raises:
    NoWorkingCameraError: If the requested device cannot deliver frames, or if
      auto-selection finds none.
  """
  if index is None:
    # Score and select in a single pass, keeping the winning device open. An
    # earlier version scored every device, released them all, then re-opened the
    # best one -- but that re-open can transiently fail while the platform tears
    # the previous capture graph down, and falling through to the next candidate
    # silently selected the blank infrared device the scoring existed to reject.
    best_capture = None
    best_index = -1
    best_score = -1.0
    scores: dict[int, float] = {}

    for candidate in range(probe_limit):
      capture = capture_factory(candidate)
      keep = False
      try:
        if not capture.isOpened():
          continue
        frames = _sample_frames(capture, _SCORING_FRAMES, warmup_seconds)
        if not frames:
          continue

        score = liveness_score(frames)
        scores[candidate] = score
        if score > best_score:
          if best_capture is not None:
            best_capture.release()
          best_capture, best_index, best_score = capture, candidate, score
          keep = True
      finally:
        if not keep:
          capture.release()
          time.sleep(settle_seconds)

    if best_capture is None:
      raise NoWorkingCameraError(
          f'No camera delivered frames (probed indices 0-{probe_limit - 1}).'
          ' Check that a webcam is connected and that no other application is'
          ' using it.'
      )

    readable_scores = {i: round(s, 2) for i, s in sorted(scores.items())}
    if best_score < _BLANK_SCORE_THRESHOLD:
      logging.warning(
          'Camera %d was selected but looks blank (liveness %.2f of %s). It is'
          ' probably an infrared or virtual device. Run with --list_cameras and'
          ' pass --camera explicitly.',
          best_index,
          best_score,
          readable_scores,
      )
    else:
      logging.info(
          'Using camera %d (liveness %.2f of %s).',
          best_index,
          best_score,
          readable_scores,
      )
    return FrameSource(best_capture, best_index)

  capture = capture_factory(index)
  if capture.isOpened() and (
      _read_with_warmup(capture, warmup_seconds) is not None
  ):
    logging.info('Using camera %d.', index)
    return FrameSource(capture, index)
  capture.release()

  alternatives = probe_cameras(
      probe_limit, capture_factory, warmup_seconds, settle_seconds
  )
  raise NoWorkingCameraError(
      _unavailable_message(index, alternatives)
  )


def _unavailable_message(index: int, alternatives: Sequence[int]) -> str:
  """Builds an actionable error for a camera that cannot deliver frames."""
  detail = (
      f'Camera {index} did not deliver any frames. Note that a device can open'
      ' successfully and still produce nothing -- infrared and virtual cameras'
      ' typically do, and often take index 0.'
  )
  if alternatives:
    usable = ', '.join(str(i) for i in alternatives)
    return (
        f'{detail} Working camera indices: {usable}.'
        f' Try --camera {alternatives[0]}.'
    )
  return (
      f'{detail} No other working camera was found either; check that a webcam'
      ' is connected and not in use by another application.'
  )
