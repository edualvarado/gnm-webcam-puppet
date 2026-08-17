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

r"""Headless end-to-end check of the puppeteering pipeline.

Poses GNM with known parameters, renders those poses as stand-in camera frames,
then runs the real pipeline -- detect, solve, re-render -- over them. Because
the input pose is known, the output can be checked rather than merely eyeballed.

This exercises everything the live app does except camera capture, so it is the
way to verify a change without a webcam.

Run with:

  python -m webcam_puppet.selftest --output_dir /tmp/puppet_selftest
"""

import dataclasses

from absl import app
from absl import flags
from absl import logging
from etils import epath
from gnm.shape import gnm_numpy
import imageio.v3 as iio
import numpy as np

from webcam_puppet import correspondence as correspondence_lib
from webcam_puppet import renderer as renderer_lib
from webcam_puppet import solver as solver_lib
from webcam_puppet import tracker as tracker_lib

_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir', None, 'Directory to write comparison images into.'
)
_REBUILD_CORRESPONDENCE = flags.DEFINE_boolean(
    'rebuild_correspondence', False, 'Re-derive the correspondence.'
)

_FRAME_SIZE = (640, 640)
_CACHE_PATH = (
    epath.resource_path('webcam_puppet') / 'assets' / 'correspondence.npz'
)

# Rendered stand-in frames need a photographic look for the detector to fire.
_FRAME_BASE_COLOR = (0.80, 0.66, 0.58)
_FRAME_BACKGROUND = (1.0, 1.0, 1.0)

# Yaw beyond this hides one side of the face from the camera, so the tracker's
# landmarks there are extrapolated and the recovered pose degrades.
_MAX_RELIABLE_YAW_DEGREES = 30.0
_MAX_YAW_ERROR_DEGREES = 4.0

# How much of a known deformation the pipeline is expected to reproduce.
#
# This is deliberately modest because the ceiling is the landmarker, not the
# solver. Measured against a known GNM deformation, MediaPipe's per-landmark
# displacement has roughly the right magnitude (0.89x) but only weakly agreeing
# direction: per-axis correlation with ground truth is 0.73 / 0.49 / 0.47 for
# x / y / z, and the mean cosine between true and observed displacement is 0.37.
# Fitting a 160-component basis averages much of that direction noise out, which
# is why the result reads convincingly in motion, but it caps faithful recovery
# near a quarter of the true deformation. Driving the eyes from MediaPipe's
# blendshape scores instead of landmark geometry is the most promising way to
# raise this; see README.md.
#
# Solving the same deformation from *exact* landmarks recovers ~87% (see
# solver_test.py), which is what isolates this ceiling to the landmarker.
_MIN_DEFORMATION_RECOVERY = 0.10

# Below this, a case carries no meaningful deformation to recover and the
# recovery ratio is dominated by its own denominator.
_MIN_MEANINGFUL_DEFORMATION_METRES = 1e-3

# A resting face must not register invented expression. This is the check that
# differential fitting exists to satisfy.
#
# The allowance grows with head yaw because expression and pose are not fully
# separable from landmarks alone: the neutral reference is captured frontal, so
# as the head turns, the landmarker's own pose-dependent shape drift and the
# growing self-occlusion of the far cheek leak into the expression residual.
# Measured leakage is ~0.6mm frontal and ~1.8mm at 20 degrees.
_MAX_NEUTRAL_RESIDUAL_METRES = 1.0e-3
_NEUTRAL_RESIDUAL_ALLOWANCE_PER_DEGREE = 0.05e-3


@dataclasses.dataclass(frozen=True)
class _Case:
  """One end-to-end check.

  Attributes:
    expression: Ground-truth expression coefficients.
    yaw_degrees: Ground-truth head yaw.
    min_recovery: Fraction of the deformation that must be reproduced, or None
      when the case is not expected to recover deformation at all.
    note: Why min_recovery is what it is.
  """

  expression: np.ndarray
  yaw_degrees: float = 0.0
  min_recovery: float | None = None
  note: str = ''


def _test_cases(gnm: gnm_numpy.GNM) -> dict[str, _Case]:
  """Builds named cases spanning the visible expression regions."""
  rng = np.random.default_rng(0)
  blocks = solver_lib.expression_region_blocks(gnm)
  lower_face = blocks['lower_face_region']
  left_eye = blocks['left_eye_region']
  right_eye = blocks['right_eye_region']

  def expression(*regions) -> np.ndarray:
    coefficients = np.zeros(gnm.expression_dim, dtype=np.float32)
    for indices, amplitude, count in regions:
      coefficients[indices[:count]] = rng.normal(0.0, amplitude, count)
    return coefficients

  neutral = np.zeros(gnm.expression_dim, dtype=np.float32)
  return {
      'neutral': _Case(neutral, note='resting face must invent no expression'),
      'yaw_20': _Case(
          neutral, yaw_degrees=20.0, note='pose must not leak into expression'
      ),
      'mouth': _Case(
          expression((lower_face, 2.0, 8)),
          min_recovery=_MIN_DEFORMATION_RECOVERY,
          note='the region landmarks resolve best',
      ),
      'mouth_and_yaw': _Case(
          expression((lower_face, 2.0, 8)),
          yaw_degrees=15.0,
          min_recovery=_MIN_DEFORMATION_RECOVERY,
          note='expression must survive simultaneous head rotation',
      ),
      # Eyelid motion is not recovered from landmark geometry: the deformation
      # is sub-millimetre and MediaPipe's eyelid landmarks do not track it
      # faithfully enough. Asserting only that it stays bounded documents the
      # gap rather than hiding it. Driving eyes from MediaPipe's blendshape
      # scores is the fix; see README.md.
      'eyes': _Case(
          expression((left_eye, 2.0, 6), (right_eye, 2.0, 6)),
          note='known gap: eyelids need blendshapes, not landmarks',
      ),
      'everything': _Case(
          expression(
              (lower_face, 1.8, 8), (left_eye, 1.5, 6), (right_eye, 1.5, 6)
          ),
          yaw_degrees=-12.0,
          min_recovery=_MIN_DEFORMATION_RECOVERY,
          note='combined motion, dominated by the mouth',
      ),
  }


def main(argv):
  del argv

  gnm = gnm_numpy.GNM.from_local(
      version=gnm_numpy.GNMMajorVersion.V3,
      variant=gnm_numpy.GNMVariant.HEAD,
  )
  head_joint = list(gnm.joint_names).index('head')

  frame_camera = renderer_lib.Camera.fit_to_mesh(
      gnm.template_vertex_positions, _FRAME_SIZE
  )
  frame_renderer = renderer_lib.MeshRenderer(
      gnm.triangles,
      frame_camera,
      background=_FRAME_BACKGROUND,
      base_color=_FRAME_BASE_COLOR,
  )

  with tracker_lib.FaceTracker(video_mode=False) as tracker:
    correspondence = correspondence_lib.load_or_build_correspondence(
        gnm,
        tracker.landmarks_only,
        cache_path=_CACHE_PATH,
        rebuild=_REBUILD_CORRESPONDENCE.value,
    )
    logging.info(
        'Correspondence: %d landmarks, %d rigid anchors.',
        len(correspondence.vertex_indices),
        int(correspondence.rigid.sum()),
    )

    tracked = correspondence.vertex_indices
    neutral_mesh = gnm.template_vertex_positions
    identity = np.zeros(gnm.identity_dim, dtype=np.float32)
    translation = np.zeros(3, dtype=np.float32)

    output_dir = None
    if _OUTPUT_DIR.value:
      output_dir = epath.Path(_OUTPUT_DIR.value)
      output_dir.mkdir(parents=True, exist_ok=True)

    print()
    header = (
        f'{'case':16s} {'deform':>9s} {'residual':>9s} {'recovered':>10s}'
        f' {'yaw true':>9s} {'yaw fit':>8s} {'RMSE':>8s}'
    )
    print(header)
    print('-' * len(header))

    failures = []
    for name, case in _test_cases(gnm).items():
      expression = case.expression
      yaw_degrees = case.yaw_degrees
      rotations = np.zeros((gnm.num_joints, 3), dtype=np.float32)
      rotations[head_joint, 1] = np.deg2rad(yaw_degrees)
      truth = gnm(identity, expression, rotations, translation)

      frame = frame_renderer.render(truth)
      observation = tracker.detect(frame)
      if observation is None:
        failures.append(f'{name}: no face detected in stand-in frame')
        continue

      solver = solver_lib.LandmarkSolver(gnm, correspondence, smoothing=0.0)
      parameters = solver.solve(
          observation.landmarks, _FRAME_SIZE[1], _FRAME_SIZE[0]
      )
      reconstruction = gnm(
          parameters.identity,
          parameters.expression,
          parameters.rotations,
          parameters.translation,
      )

      # Compare in the neutral frame so pose error does not mask shape error.
      unposed_truth = gnm(
          identity, expression, np.zeros_like(rotations), translation
      )
      unposed_fit = gnm(
          parameters.identity,
          parameters.expression,
          np.zeros_like(rotations),
          translation,
      )
      deformation = np.linalg.norm(
          unposed_truth[tracked] - neutral_mesh[tracked], axis=-1
      ).mean()
      residual = np.linalg.norm(
          unposed_fit[tracked] - unposed_truth[tracked], axis=-1
      ).mean()

      has_deformation = deformation > _MIN_MEANINGFUL_DEFORMATION_METRES
      recovered = 1.0 - residual / deformation if has_deformation else None
      recovered_text = (
          f'{recovered * 100:8.1f}%' if recovered is not None else '       --'
      )

      fitted_yaw = np.rad2deg(parameters.rotations[head_joint, 1])
      print(
          f'{name:16s} {deformation * 1000:7.2f}mm {residual * 1000:7.2f}mm'
          f' {recovered_text} {yaw_degrees:8.1f}d {fitted_yaw:7.1f}d'
          f' {parameters.landmark_rmse * 1000:6.2f}mm'
      )

      if abs(yaw_degrees) <= _MAX_RELIABLE_YAW_DEGREES:
        yaw_error = abs(fitted_yaw - yaw_degrees)
        if yaw_error > _MAX_YAW_ERROR_DEGREES:
          failures.append(
              f'{name}: yaw error {yaw_error:.1f} degrees exceeds'
              f' {_MAX_YAW_ERROR_DEGREES}'
          )

      if case.min_recovery is not None:
        if recovered is None:
          failures.append(
              f'{name}: expected recoverable deformation but the case produced'
              f' only {deformation * 1000:.2f}mm'
          )
        elif recovered < case.min_recovery:
          failures.append(
              f'{name}: recovered {recovered * 100:.0f}% of deformation, below'
              f' {case.min_recovery * 100:.0f}% ({case.note})'
          )
      else:
        # Nothing is expected to be recovered, so the residual must simply stay
        # bounded: this catches invented expression on a resting face and
        # runaway coefficients on the known-weak eyelid path alike.
        allowance = (
            _MAX_NEUTRAL_RESIDUAL_METRES
            + _NEUTRAL_RESIDUAL_ALLOWANCE_PER_DEGREE * abs(yaw_degrees)
            + deformation
        )
        if residual > allowance:
          failures.append(
              f'{name}: {residual * 1000:.2f}mm residual exceeds the'
              f' {allowance * 1000:.2f}mm allowed ({case.note})'
          )

      if output_dir is not None:
        rendered_fit = frame_renderer.render(reconstruction)
        comparison = np.hstack([frame, rendered_fit])
        iio.imwrite(str(output_dir / f'{name}.png'), comparison)

    print()
    if failures:
      for failure in failures:
        logging.error('%s', failure)
      raise SystemExit(f'{len(failures)} check(s) failed.')

    if output_dir is not None:
      logging.info('Wrote comparison images to %s', output_dir)
    logging.info('All checks passed.')


if __name__ == '__main__':
  app.run(main)
