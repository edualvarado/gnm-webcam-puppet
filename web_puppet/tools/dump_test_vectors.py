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

"""Dumps golden (parameters -> vertices) pairs for the TypeScript test.

`reference_model.py` is already checked against `gnm.shape`, so freezing its
output here extends that guarantee across the language boundary: the browser
forward pass is correct if it reproduces these, without Python ever running
next to it.

Cases deliberately span the axes that break independently -- an identity-only
draw exercises the identity basis and the joint basis together, an
expression-only draw isolates the expression basis, and a posed draw is the
only one that exercises forward kinematics and skinning at all.

Usage:
  python -m web_puppet.tools.dump_test_vectors
"""

from collections.abc import Sequence
import json
import os

from absl import app
from absl import flags
from absl import logging
import numpy as np

from web_puppet.tools import reference_model

_ASSET_DIR = flags.DEFINE_string(
    'asset_dir',
    'web_puppet/public/assets',
    'Directory holding the exported gnm_head.bin and gnm_head.json.',
)
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    'web_puppet/testdata',
    'Directory to write golden.json and golden.bin into.',
)


def build_cases(
    model: reference_model.ReferenceModel,
) -> list[dict[str, object]]:
  """Builds the parameter sets to freeze.

  Args:
    model: The loaded reference model.

  Returns:
    A list of named parameter dictionaries.
  """
  rng = np.random.default_rng(20260801)
  zeros_rotations = np.zeros((model.num_joints, 3), dtype=np.float32)

  def case(name, identity=None, expression=None, rotations=None,
           translation=None):
    return {
        'name': name,
        'identity': (
            np.zeros(model.identity_dim, np.float32)
            if identity is None else identity
        ),
        'expression': (
            np.zeros(model.expression_dim, np.float32)
            if expression is None else expression
        ),
        'rotations': zeros_rotations if rotations is None else rotations,
        'translation': (
            np.zeros(3, np.float32) if translation is None else translation
        ),
    }

  identity = rng.normal(0.0, 1.0, model.identity_dim).astype(np.float32)
  expression = rng.normal(0.0, 1.0, model.expression_dim).astype(np.float32)

  # A yaw-dominant rotation on the head joint, plus a small neck tilt, is what
  # actually distinguishes a correct joint hierarchy from a flat one: get the
  # parenting wrong and the neck contribution vanishes.
  rotations = np.zeros((model.num_joints, 3), dtype=np.float32)
  rotations[0] = (0.05, 0.02, -0.03)
  rotations[1] = (0.10, 0.35, 0.05)
  rotations[2] = (0.02, -0.08, 0.0)
  rotations[3] = (0.02, 0.08, 0.0)

  return [
      case('template'),
      case('identity_only', identity=identity),
      case('expression_only', expression=expression),
      case('posed', rotations=rotations,
           translation=np.asarray([0.01, -0.02, 0.03], np.float32)),
      case('combined', identity=identity, expression=expression,
           rotations=rotations,
           translation=np.asarray([-0.005, 0.01, 0.02], np.float32)),
  ]


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError(f'Unexpected arguments: {argv[1:]}')

  model = reference_model.ReferenceModel.load(_ASSET_DIR.value)
  cases = build_cases(model)

  blob = bytearray()
  entries = []
  for spec in cases:
    vertices = model(
        spec['identity'],
        spec['expression'],
        spec['rotations'],
        spec['translation'],
    ).astype(np.float32)

    entries.append({
        'name': spec['name'],
        'identity': spec['identity'].tolist(),
        'expression': spec['expression'].tolist(),
        'rotations': spec['rotations'].tolist(),
        'translation': spec['translation'].tolist(),
        'byteOffset': len(blob),
        'byteLength': vertices.nbytes,
    })
    blob.extend(np.ascontiguousarray(vertices).tobytes())

  os.makedirs(_OUTPUT_DIR.value, exist_ok=True)
  binary_path = os.path.join(_OUTPUT_DIR.value, 'golden.bin')
  manifest_path = os.path.join(_OUTPUT_DIR.value, 'golden.json')

  with open(binary_path, 'wb') as handle:
    handle.write(blob)
  with open(manifest_path, 'w', encoding='utf-8') as handle:
    json.dump(
        {
            'vertexCount': model.num_vertices,
            'identityDim': model.identity_dim,
            'expressionDim': model.expression_dim,
            'cases': entries,
        },
        handle,
        indent=2,
    )

  logging.info('Wrote %s', manifest_path)
  logging.info('Wrote %s (%.2f MB)', binary_path, len(blob) / 1e6)
  for entry in entries:
    name = entry['name']
    print(f'  {name}')
  print(f'{len(entries)} cases, {len(blob) / 1e6:.2f} MB')


if __name__ == '__main__':
  app.run(main)
