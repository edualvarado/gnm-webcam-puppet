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

"""Tests for camera opening and frame reading.

Uses fake capture devices so the behaviour that matters -- preferring a live
colour camera over one that merely delivers frames -- is exercised without real
hardware.
"""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from webcam_puppet import camera_source

# Fakes need no real warm-up; keep the unreadable-device cases fast.
_TEST_WARMUP = 0.05


class _FakeCapture:
  """Stands in for cv2.VideoCapture.

  Attributes:
    released: Whether release() has been called.
  """

  def __init__(
      self,
      opened: bool = True,
      readable_after: int = 0,
      total_reads: int | None = None,
      blank: bool = False,
  ):
    """Builds a fake capture.

    Args:
      opened: What isOpened() reports.
      readable_after: Number of initial read() calls that fail, modelling
        warm-up.
      total_reads: Successful reads to allow before failing forever, or None for
        unlimited.
      blank: If True, emit dark greyscale frames that never change, imitating an
        infrared sensor. Otherwise emit varying colour frames.
    """
    self._opened = opened
    self._readable_after = readable_after
    self._total_reads = total_reads
    self._blank = blank
    self._read_calls = 0
    self._successful_reads = 0
    self.released = False

  def isOpened(self) -> bool:  # pylint: disable=invalid-name
    """Matches the OpenCV method name."""
    return self._opened

  def read(self):
    """Returns (ok, frame) like cv2.VideoCapture.read."""
    self._read_calls += 1
    if self._read_calls <= self._readable_after:
      return False, None
    if self._total_reads is not None and (
        self._successful_reads >= self._total_reads
    ):
      return False, None
    self._successful_reads += 1
    if self._blank:
      # Dark, greyscale, identical every frame.
      return True, np.full((8, 8, 3), 20, dtype=np.uint8)

    # Distinct per-channel values plus per-frame variation.
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    frame[:, :, 0] = 40 + 10 * self._successful_reads
    frame[:, :, 1] = 120
    frame[:, :, 2] = 200
    return True, frame

  def release(self) -> None:
    """Records release."""
    self.released = True


def _factory(devices: dict[int, _FakeCapture]):
  """Builds a capture factory over a fixed device map."""

  def factory(index: int) -> _FakeCapture:
    return devices.get(index) or _FakeCapture(opened=False)

  return factory


class OpenCameraTest(parameterized.TestCase):

  def test_auto_selects_first_working_device(self):
    devices = {0: _FakeCapture(), 1: _FakeCapture()}

    source = camera_source.open_camera(
        None,
        capture_factory=_factory(devices),
        probe_limit=3,
        warmup_seconds=_TEST_WARMUP,
        settle_seconds=0.0,
    )

    self.assertEqual(source.index, 0)

  def test_auto_skips_device_that_opens_but_never_reads(self):
    # isOpened() is useless as a readiness check: this device opens and then
    # never yields a frame.
    devices = {
        0: _FakeCapture(opened=True, total_reads=0),
        1: _FakeCapture(opened=True),
    }

    source = camera_source.open_camera(
        None,
        capture_factory=_factory(devices),
        probe_limit=3,
        warmup_seconds=_TEST_WARMUP,
        settle_seconds=0.0,
    )

    self.assertEqual(source.index, 1)

  def test_auto_tolerates_warmup_failures(self):
    # A real camera can fail its first few reads and work moments later, so the
    # window here must span several poll intervals rather than the minimal one
    # the other cases use.
    devices = {0: _FakeCapture(readable_after=3)}

    source = camera_source.open_camera(
        None,
        capture_factory=_factory(devices),
        probe_limit=2,
        warmup_seconds=1.0,
        settle_seconds=0.0,
    )

    self.assertEqual(source.index, 0)

  def test_auto_raises_when_nothing_works(self):
    devices = {0: _FakeCapture(opened=True, total_reads=0)}

    with self.assertRaisesRegex(
        camera_source.NoWorkingCameraError, 'No camera delivered frames'
    ):
      camera_source.open_camera(
          None,
          capture_factory=_factory(devices),
          probe_limit=2,
          warmup_seconds=_TEST_WARMUP,
          settle_seconds=0.0,
      )

  def test_explicit_index_is_honoured(self):
    devices = {0: _FakeCapture(), 1: _FakeCapture()}

    source = camera_source.open_camera(
        1,
        capture_factory=_factory(devices),
        probe_limit=3,
        warmup_seconds=_TEST_WARMUP,
        settle_seconds=0.0,
    )

    self.assertEqual(source.index, 1)

  def test_explicit_unreadable_index_suggests_working_one(self):
    devices = {
        0: _FakeCapture(opened=True, total_reads=0),
        1: _FakeCapture(opened=True),
    }

    with self.assertRaisesRegex(
        camera_source.NoWorkingCameraError, r'--camera 1'
    ):
      camera_source.open_camera(
          0,
          capture_factory=_factory(devices),
          probe_limit=3,
          warmup_seconds=_TEST_WARMUP,
          settle_seconds=0.0,
      )

  def test_explicit_missing_index_reports_no_alternatives(self):
    with self.assertRaisesRegex(
        camera_source.NoWorkingCameraError, 'No other working camera'
    ):
      camera_source.open_camera(
          2,
          capture_factory=_factory({}),
          probe_limit=3,
          warmup_seconds=_TEST_WARMUP,
          settle_seconds=0.0,
      )

  def test_rejected_devices_are_released(self):
    unreadable = _FakeCapture(opened=True, total_reads=0)
    devices = {0: unreadable, 1: _FakeCapture()}

    camera_source.open_camera(
        None,
        capture_factory=_factory(devices),
        probe_limit=3,
        warmup_seconds=_TEST_WARMUP,
        settle_seconds=0.0,
    )

    self.assertTrue(unreadable.released)


class LivenessSelectionTest(parameterized.TestCase):

  def test_prefers_live_colour_camera_over_blank_device(self):
    # The reported bug: an infrared sensor at index 0 delivers frames, so
    # "did a frame arrive" selects it and the app shows no webcam. The real
    # camera must win even though it enumerates later.
    devices = {0: _FakeCapture(blank=True), 1: _FakeCapture()}

    source = camera_source.open_camera(
        None,
        capture_factory=_factory(devices),
        probe_limit=3,
        warmup_seconds=_TEST_WARMUP,
        settle_seconds=0.0,
    )

    self.assertEqual(source.index, 1)

  def test_selects_blank_device_only_when_it_is_the_sole_option(self):
    devices = {0: _FakeCapture(blank=True)}

    source = camera_source.open_camera(
        None,
        capture_factory=_factory(devices),
        probe_limit=3,
        warmup_seconds=_TEST_WARMUP,
        settle_seconds=0.0,
    )

    self.assertEqual(source.index, 0)

  def test_blank_frames_score_far_below_live_frames(self):
    blank = _FakeCapture(blank=True)
    live = _FakeCapture()
    blank_frames = [blank.read()[1] for _ in range(4)]
    live_frames = [live.read()[1] for _ in range(4)]

    blank_score = camera_source.liveness_score(blank_frames)
    live_score = camera_source.liveness_score(live_frames)

    self.assertLess(blank_score, 1.0)
    self.assertGreater(live_score, 10.0 * max(blank_score, 1e-6))

  def test_liveness_of_no_frames_is_zero(self):
    self.assertEqual(camera_source.liveness_score([]), 0.0)

  def test_single_frame_scores_on_colour_alone(self):
    live = _FakeCapture()
    single = [live.read()[1]]

    self.assertGreater(camera_source.liveness_score(single), 0.0)

  def test_selected_source_is_still_usable(self):
    # Regression guard: scoring once released every device and then re-opened
    # the winner, which could transiently fail and fall through to the blank
    # device. The returned source must already be live.
    devices = {0: _FakeCapture(blank=True), 1: _FakeCapture()}

    source = camera_source.open_camera(
        None,
        capture_factory=_factory(devices),
        probe_limit=3,
        warmup_seconds=_TEST_WARMUP,
        settle_seconds=0.0,
    )

    self.assertIsNotNone(source.read())


class ScoreCamerasTest(parameterized.TestCase):

  def test_scores_only_readable_devices(self):
    devices = {
        0: _FakeCapture(opened=True, total_reads=0),
        1: _FakeCapture(),
    }

    scores = camera_source.score_cameras(
        limit=3,
        capture_factory=_factory(devices),
        warmup_seconds=_TEST_WARMUP,
        settle_seconds=0.0,
    )

    self.assertEqual(list(scores), [1])

  def test_blank_device_is_scored_but_ranked_last(self):
    devices = {0: _FakeCapture(blank=True), 1: _FakeCapture()}

    scores = camera_source.score_cameras(
        limit=3,
        capture_factory=_factory(devices),
        warmup_seconds=_TEST_WARMUP,
        settle_seconds=0.0,
    )

    self.assertCountEqual(list(scores), [0, 1])
    self.assertLess(scores[0], scores[1])


class ProbeCamerasTest(parameterized.TestCase):

  def test_reports_only_readable_devices(self):
    devices = {
        0: _FakeCapture(opened=True, total_reads=0),
        1: _FakeCapture(),
        3: _FakeCapture(),
    }

    working = camera_source.probe_cameras(
        limit=4,
        capture_factory=_factory(devices),
        warmup_seconds=_TEST_WARMUP,
        settle_seconds=0.0,
    )

    self.assertEqual(working, [1, 3])

  def test_returns_empty_when_no_devices(self):
    working = camera_source.probe_cameras(
        limit=3,
        capture_factory=_factory({}),
        warmup_seconds=_TEST_WARMUP,
        settle_seconds=0.0,
    )

    self.assertEmpty(working)

  def test_releases_every_probed_device(self):
    devices = {0: _FakeCapture(), 1: _FakeCapture()}

    camera_source.probe_cameras(
        limit=2,
        capture_factory=_factory(devices),
        warmup_seconds=_TEST_WARMUP,
        settle_seconds=0.0,
    )

    self.assertTrue(all(device.released for device in devices.values()))


class FrameSourceTest(parameterized.TestCase):

  def test_read_returns_frames(self):
    source = camera_source.FrameSource(_FakeCapture(), index=0)

    frame = source.read()

    self.assertIsNotNone(frame)
    self.assertEqual(frame.shape, (8, 8, 3))

  def test_read_survives_transient_failure(self):
    # A single dropped frame must not end a live session.
    capture = _FakeCapture(readable_after=2)
    source = camera_source.FrameSource(capture, index=0)

    self.assertIsNotNone(source.read())

  def test_read_returns_none_once_camera_dies(self):
    capture = _FakeCapture(total_reads=1)
    source = camera_source.FrameSource(capture, index=0)

    self.assertIsNotNone(source.read())
    self.assertIsNone(source.read())

  def test_context_manager_releases(self):
    capture = _FakeCapture()

    with camera_source.FrameSource(capture, index=0):
      pass

    self.assertTrue(capture.released)


if __name__ == '__main__':
  absltest.main()
