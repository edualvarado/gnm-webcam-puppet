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

r"""Drives the GNM head from a webcam in real time.

Run with:

  python -m webcam_puppet.app

Keys:
  n  capture identity from the current frame (hold a neutral face first)
  r  reset identity back to the template
  m  mirror the camera preview
  q  quit
"""

import time

from absl import app
from absl import flags
from absl import logging
import cv2
from etils import epath
from gnm.shape import gnm_numpy
import numpy as np
import numpy.typing as npt

from webcam_puppet import camera_source as camera_lib
from webcam_puppet import correspondence as correspondence_lib
from webcam_puppet import renderer as renderer_lib
from webcam_puppet import solver as solver_lib
from webcam_puppet import tracker as tracker_lib

_CAMERA = flags.DEFINE_integer(
    'camera',
    None,
    'Camera device index. Omit to auto-select the first device that actually'
    ' delivers frames, which skips infrared and virtual cameras that open'
    ' successfully but produce no picture.',
)
_LIST_CAMERAS = flags.DEFINE_boolean(
    'list_cameras', False, 'List working camera indices and exit.'
)
_RENDER_SIZE = flags.DEFINE_integer(
    'render_size', 720, 'Edge length in pixels of the rendered head.'
)
_SMOOTHING = flags.DEFINE_float(
    'smoothing',
    0.4,
    'Temporal smoothing in [0, 1). Higher is steadier but laggier.',
)
_EXPRESSION_GAIN = flags.DEFINE_float(
    'expression_gain',
    2.0,
    'Exaggerates solved expression. Landmark tracking recovers only ~25% of'
    ' expression magnitude, so amplifying makes motion read on screen; it also'
    ' amplifies direction error, so past ~3 the face turns lumpy. 1.0 is the'
    ' unexaggerated fit.',
)
_MIRROR = flags.DEFINE_boolean(
    'mirror', True, 'Mirror the camera preview so it reads like a mirror.'
)
_REBUILD_CORRESPONDENCE = flags.DEFINE_boolean(
    'rebuild_correspondence',
    False,
    'Re-derive the landmark-to-vertex correspondence instead of using the'
    ' cache.',
)
_RECORD = flags.DEFINE_string(
    'record', None, 'Optional path to write the composited view as an MP4.'
)

_WINDOW_NAME = 'GNM webcam puppet'
_CACHE_PATH = (
    epath.resource_path('webcam_puppet') / 'assets' / 'correspondence.npz'
)

_HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
_HUD_COLOR = (235, 235, 235)
_HUD_WARN_COLOR = (90, 170, 250)


def _draw_hud(
    image: npt.NDArray[np.uint8],
    lines: list[tuple[str, tuple[int, int, int]]],
) -> None:
  """Draws left-aligned HUD text onto a BGR image in place."""
  for row, (text, color) in enumerate(lines):
    cv2.putText(
        image,
        text,
        (12, 26 + row * 22),
        _HUD_FONT,
        0.55,
        color,
        1,
        cv2.LINE_AA,
    )


def _composite(
    preview_bgr: npt.NDArray[np.uint8],
    render_rgb: npt.NDArray[np.uint8],
) -> npt.NDArray[np.uint8]:
  """Stacks the camera preview and the rendered head side by side.

  The preview is letterboxed into a panel the same size as the render rather
  than matched on height, so that a wide camera cannot take the majority of the
  window away from the head.
  """
  render_bgr = cv2.cvtColor(render_rgb, cv2.COLOR_RGB2BGR)
  height, width = render_bgr.shape[:2]

  scale = min(width / preview_bgr.shape[1], height / preview_bgr.shape[0])
  fitted = cv2.resize(
      preview_bgr,
      (int(round(preview_bgr.shape[1] * scale)),
       int(round(preview_bgr.shape[0] * scale))),
      interpolation=cv2.INTER_AREA,
  )

  panel = np.zeros_like(render_bgr)
  top = (height - fitted.shape[0]) // 2
  left = (width - fitted.shape[1]) // 2
  panel[top:top + fitted.shape[0], left:left + fitted.shape[1]] = fitted
  return np.hstack([panel, render_bgr])


def main(argv):
  del argv

  if _LIST_CAMERAS.value:
    scores = camera_lib.score_cameras()
    if not scores:
      print('No camera delivered any frames.')
      return
    print('index  liveness  assessment')
    for index, score in sorted(scores.items()):
      # Infrared and virtual devices deliver frames but score near zero: dark,
      # greyscale and static.
      assessment = (
          'looks like a real webcam'
          if score >= 1.0
          else 'delivers frames but looks blank (infrared or virtual device?)'
      )
      print(f'{index:>5}  {score:>8.2f}  {assessment}')
    best = max(scores.items(), key=lambda item: item[1])[0]
    print()
    print(f'Auto-selection would use --camera {best}.')
    return

  logging.info('Loading GNM head model ...')
  gnm = gnm_numpy.GNM.from_local(
      version=gnm_numpy.GNMMajorVersion.V3,
      variant=gnm_numpy.GNMVariant.HEAD,
  )

  tracker = tracker_lib.FaceTracker(video_mode=True)

  logging.info('Resolving landmark correspondence ...')
  # Correspondence is derived from a still render, so it needs an IMAGE-mode
  # detector rather than the VIDEO-mode one used for the live loop.
  with tracker_lib.FaceTracker(video_mode=False) as calibration_tracker:
    correspondence = correspondence_lib.load_or_build_correspondence(
        gnm,
        calibration_tracker.landmarks_only,
        cache_path=_CACHE_PATH,
        rebuild=_REBUILD_CORRESPONDENCE.value,
    )
  logging.info(
      'Correspondence: %d landmarks, %d rigid anchors.',
      len(correspondence.vertex_indices),
      int(correspondence.rigid.sum()),
  )

  solver = solver_lib.LandmarkSolver(
      gnm,
      correspondence,
      smoothing=_SMOOTHING.value,
      expression_gain=_EXPRESSION_GAIN.value,
  )

  render_size = (_RENDER_SIZE.value, _RENDER_SIZE.value)
  # Frame the head rather than the whole mesh. The shoulder plate is the widest
  # and lowest part of the bust, so framing all of it pushes the face into the
  # middle third of the window, which is the opposite of what this view is for.
  head_joint = list(gnm.joint_names).index('head')
  camera = renderer_lib.Camera.fit_to_mesh(
      gnm.template_vertex_positions[gnm.skinning_weights[head_joint] >= 0.5],
      render_size,
      # The camera is fixed to the neutral head, so leave room for the posed
      # head to turn and translate without touching the edge of the frame.
      fill_factor=0.75,
  )
  renderer = renderer_lib.MeshRenderer(gnm.triangles, camera)

  # An auto-sized window pins the view to render_size; a normal one lets the
  # composite be dragged to any size, and keeps its aspect ratio while doing so.
  cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_NORMAL)
  cv2.setWindowProperty(
      _WINDOW_NAME, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO
  )
  cv2.resizeWindow(_WINDOW_NAME, 2 * _RENDER_SIZE.value, _RENDER_SIZE.value)

  frame_source = camera_lib.open_camera(_CAMERA.value)

  writer = None
  mirror = _MIRROR.value
  pending_identity_capture = False
  frame_index = 0
  smoothed_fps = 0.0
  last_time = time.perf_counter()

  neutral_expression = np.zeros(gnm.expression_dim, dtype=np.float32)
  neutral_rotations = np.zeros((gnm.num_joints, 3), dtype=np.float32)
  neutral_translation = np.zeros(3, dtype=np.float32)

  logging.info('Running. Press n for identity capture, q to quit.')
  try:
    while True:
      frame_bgr = frame_source.read()
      if frame_bgr is None:
        logging.warning('Camera stopped delivering frames; stopping.')
        break

      frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
      height, width = frame_rgb.shape[:2]

      # MediaPipe video mode needs strictly increasing timestamps; derive
      # them from the frame counter so they stay monotonic even if the clock
      # does not.
      timestamp_ms = int(frame_index * 1000 / 30)
      observation = tracker.detect(frame_rgb, timestamp_ms=timestamp_ms)
      frame_index += 1

      if observation is None:
        vertices = gnm(
            solver.identity,
            neutral_expression,
            neutral_rotations,
            neutral_translation,
        )
        status = [('no face detected', _HUD_WARN_COLOR)]
      else:
        if pending_identity_capture:
          solver.solve_identity(observation.landmarks, width, height)
          pending_identity_capture = False
          logging.info('Captured identity from current frame.')

        parameters = solver.solve(observation.landmarks, width, height)
        vertices = gnm(
            parameters.identity,
            parameters.expression,
            parameters.rotations,
            parameters.translation,
        )
        status = [(
            f'fit RMSE {parameters.landmark_rmse * 1000:5.2f} mm',
            _HUD_COLOR,
        )]

      render_rgb = renderer.render(vertices)

      preview = cv2.flip(frame_bgr, 1) if mirror else frame_bgr
      composited = _composite(preview, render_rgb)

      now = time.perf_counter()
      instantaneous_fps = 1.0 / max(now - last_time, 1e-6)
      last_time = now
      smoothed_fps = (
          instantaneous_fps
          if smoothed_fps == 0.0
          else 0.9 * smoothed_fps + 0.1 * instantaneous_fps
      )

      identity_state = (
          'identity: captured'
          if np.any(solver.identity)
          else 'identity: template (press n)'
      )
      _draw_hud(
          composited,
          [
              (f'{smoothed_fps:4.1f} fps', _HUD_COLOR),
              *status,
              (identity_state, _HUD_COLOR),
              ('n neutral   r reset   m mirror   q quit', _HUD_COLOR),
          ],
      )

      if _RECORD.value:
        if writer is None:
          writer = cv2.VideoWriter(
              _RECORD.value,
              cv2.VideoWriter_fourcc(*'mp4v'),
              30.0,
              (composited.shape[1], composited.shape[0]),
          )
        writer.write(composited)

      cv2.imshow(_WINDOW_NAME, composited)
      key = cv2.waitKey(1) & 0xFF
      if key in (ord('q'), 27):
        break
      elif key == ord('n'):
        pending_identity_capture = True
      elif key == ord('r'):
        solver.reset()
        logging.info('Reset identity to template.')
      elif key == ord('m'):
        mirror = not mirror
  finally:
    frame_source.release()
    if writer is not None:
      writer.release()
      logging.info('Wrote recording to %s', _RECORD.value)
    tracker.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
  app.run(main)
