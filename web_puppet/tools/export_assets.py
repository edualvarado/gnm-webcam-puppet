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

"""Exports the GNM head model to a browser-loadable asset pair.

The shipped model is 150 MB of raw float32, dominated by two bases that are
almost entirely tail: 64 of 253 identity components carry 98.9% of the
displacement energy, and 24 of 100 components carry 99% within each expression
region. Truncating on that basis and storing the bases as float16 brings the
download to roughly 20 MB, which a web app can reasonably fetch.

Output is one binary blob plus a JSON manifest of named views into it, so the
browser does a single ranged fetch and hands each slice straight to a typed
array. Bases are stored as raw float16 bits because that is exactly what a
WebGL half-float texture wants uploaded -- no decode step on the critical path.

Expression components are budgeted *per region*, never as a leading prefix of
the whole basis: the basis is region-blocked rather than variance-ordered, so
a global prefix selects eye components only and the face cannot open its mouth.

Usage:
  python -m web_puppet.tools.export_assets --output_dir web_puppet/public/assets
"""

from collections.abc import Mapping, Sequence
import json
import os
from typing import Any

from absl import app
from absl import flags
from absl import logging
from gnm.shape import gnm_numpy
import numpy as np
import numpy.typing as npt

_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    'web_puppet/public/assets',
    'Directory to write gnm_head.bin and gnm_head.json into.',
)
_IDENTITY_COMPONENTS = flags.DEFINE_integer(
    'identity_components',
    64,
    'Number of identity components to keep, in model order.',
)
_CORRESPONDENCE_PATH = flags.DEFINE_string(
    'correspondence_path',
    'webcam_puppet/assets/correspondence.npz',
    'MediaPipe landmark to GNM vertex mapping, as derived by the earlier '
    'prototype. Skipped if missing.',
)
_EXPRESSION_BUDGET = flags.DEFINE_string(
    'expression_budget',
    'left_eye_region=24,right_eye_region=24,lower_face_region=32,'
    'tongue=8,pupils=1',
    'Per-region expression component budget, as name=count pairs.',
)

# Arrays are concatenated into one blob; each view starts on a 4-byte boundary
# so a typed array can be constructed over the buffer without copying.
_ALIGNMENT = 4

_DTYPE_NAMES = {
    np.dtype(np.float32): 'float32',
    np.dtype(np.float16): 'float16',
    np.dtype(np.uint16): 'uint16',
    np.dtype(np.uint8): 'uint8',
}


def parse_budget(spec: str) -> dict[str, int]:
  """Parses a `name=count,...` budget string.

  Args:
    spec: Comma-separated `region=count` pairs.

  Returns:
    A mapping from region name to component count.

  Raises:
    ValueError: If a pair is malformed or a count is not a positive integer.
  """
  budget = {}
  for pair in spec.split(','):
    if not pair.strip():
      continue
    name, separator, count = pair.partition('=')
    if not separator:
      raise ValueError(f'Malformed budget entry {pair!r}, expected name=count.')
    value = int(count)
    if value <= 0:
      raise ValueError(f'Budget for {name!r} must be positive, got {value}.')
    budget[name.strip()] = value
  return budget


def region_of(name: str) -> str:
  """Returns the region a basis component name belongs to.

  Component names are `<region>_<index>`, e.g. `lower_face_region_007`.

  Args:
    name: A component name from `expression_names`.

  Returns:
    The region prefix.
  """
  return name.rsplit('_', 1)[0]


def select_expression_components(
    expression_names: Sequence[str],
    budget: Mapping[str, int],
) -> npt.NDArray[np.int32]:
  """Chooses which expression components to keep, per region.

  Args:
    expression_names: All component names, in model order.
    budget: How many components to keep from each region.

  Returns:
    Indices into the full expression basis, ascending.

  Raises:
    ValueError: If the budget names a region the model does not have.
  """
  regions: dict[str, list[int]] = {}
  for index, name in enumerate(expression_names):
    regions.setdefault(region_of(name), []).append(index)

  unknown = set(budget) - set(regions)
  if unknown:
    raise ValueError(
        f'Budget names regions not in the model: {sorted(unknown)}. '
        f'Available: {sorted(regions)}.'
    )

  kept: list[int] = []
  for region, indices in regions.items():
    count = budget.get(region, 0)
    if count > len(indices):
      logging.warning(
          'Budget for %s is %d but the region only has %d components; '
          'keeping all of them.',
          region,
          count,
          len(indices),
      )
    kept.extend(indices[:count])
  return np.asarray(sorted(kept), dtype=np.int32)


def component_ids(gnm: gnm_numpy.GNM) -> npt.NDArray[np.uint8]:
  """Assigns each vertex to one of the model's mesh components.

  The six component vertex groups are binary and partition the mesh, so an
  argmax recovers the assignment exactly.

  Args:
    gnm: The loaded model.

  Returns:
    Per-vertex component index into `mesh_component_names`, (V,) uint8.

  Raises:
    ValueError: If some vertex belongs to no component.
  """
  weights = np.stack(
      [np.asarray(gnm.vertex_group(name)) for name in gnm.mesh_component_names]
  )
  if not np.all(weights.max(axis=0) > 0):
    raise ValueError('Some vertices belong to no mesh component.')
  return weights.argmax(axis=0).astype(np.uint8)


def quad_edges(quads: npt.NDArray[np.integer]) -> npt.NDArray[np.uint16]:
  """Derives the unique undirected edge list of the quad topology.

  The quad wireframe is the readable one -- the triangulated mesh adds a
  diagonal across every quad, which reads as noise rather than as edge flow.

  Args:
    quads: Quad vertex indices, (Q, 4).

  Returns:
    Unique vertex index pairs, (E, 2) uint16.
  """
  tail = quads.ravel()
  head = np.roll(quads, -1, axis=1).ravel()
  pairs = np.sort(np.stack([tail, head], axis=1), axis=1)
  return np.unique(pairs, axis=0).astype(np.uint16)


def normal_adjacency(
    triangles: npt.NDArray[np.integer],
    vertex_count: int,
) -> tuple[npt.NDArray[np.uint16], npt.NDArray[np.uint8], int]:
  """Builds, per vertex, the incident triangles needed to rebuild its normal.

  A vertex shader cannot see a vertex's neighbours, so a GPU implementation of
  smooth normals has to be handed the adjacency explicitly.

  For each incident triangle this stores the *cyclically following two*
  vertices. That choice is what makes the GPU result exact rather than
  approximate: the cross product of two edge vectors taken from any corner of
  a triangle in cyclic order is the same vector, so `cross(p - v, q - v)`
  reproduces the CPU's `cross(b - a, c - a)` face normal identically,
  including its magnitude -- which is twice the triangle area and therefore
  the area weighting.

  Args:
    triangles: Triangle vertex indices, (T, 3).
    vertex_count: Number of mesh vertices.

  Returns:
    A tuple of the padded (V, max_corners, 2) neighbour pairs, the per-vertex
    incident triangle count, and max_corners.
  """
  incident: list[list[tuple[int, int]]] = [[] for _ in range(vertex_count)]
  for triangle in triangles:
    a, b, c = int(triangle[0]), int(triangle[1]), int(triangle[2])
    incident[a].append((b, c))
    incident[b].append((c, a))
    incident[c].append((a, b))

  max_corners = max(len(entry) for entry in incident)
  pairs = np.zeros((vertex_count, max_corners, 2), dtype=np.uint16)
  counts = np.zeros(vertex_count, dtype=np.uint8)
  for vertex, entry in enumerate(incident):
    counts[vertex] = len(entry)
    for slot, (p, q) in enumerate(entry):
      pairs[vertex, slot] = (p, q)
  return pairs, counts, max_corners


def cheek_puff_corrective(
    gnm: gnm_numpy.GNM,
    amplitude: float = 0.012,
    radius: float = 0.035,
) -> npt.NDArray[np.float32]:
  """Builds a cheek-inflation shape, which the GNM basis does not contain.

  GNM is a statistical model of face scans, and puffed cheeks are not in that
  distribution: across all 383 expression components the strongest moves the
  cheek region 0.236 mm outward per unit, so even driven to the solver's +/-3
  clamp it reaches 0.7 mm against the 8-12 mm a real puff spans. The median
  component moves it 0.000 mm. No amount of better tracking recovers a
  direction the basis does not have, so the shape is authored here instead.

  Authored, but not hand-sculpted: the region comes from the model's own vertex
  groups -- skin between the lower teeth and the eyes, lateral enough that its
  normal points sideways rather than forward -- and the displacement is along
  each vertex's normal with a smooth compact falloff about each cheek's centre.

  This is deliberately *not* GNM. It is a corrective layer on top, the way a
  production rig handles what a PCA basis lacks, and the manifest names it
  separately so nothing mistakes it for model output.

  Args:
    gnm: The loaded model.
    amplitude: Peak outward displacement in metres, at unit weight.
    radius: Falloff radius about each cheek centre, in metres.

  Returns:
    Per-vertex displacement at unit weight, (V, 3) float32.
  """
  vertices = np.asarray(gnm.template_vertex_positions, dtype=np.float64)
  skin = np.asarray(gnm.vertex_group_indices('skin'))
  normals = _vertex_normals(vertices, np.asarray(gnm.triangles))

  eyes = np.concatenate([
      np.asarray(gnm.vertex_group_indices(name))
      for name in ('left_eye', 'right_eye')
  ])
  lower_teeth = np.asarray(gnm.vertex_group_indices('lower_teeth_and_gums'))
  low = vertices[lower_teeth, 1].min()
  high = vertices[eyes, 1].min()

  sideways = np.abs(normals[:, 0]) > 0.5
  band = skin[
      sideways[skin]
      & (vertices[skin, 1] > low)
      & (vertices[skin, 1] < high)
      & (vertices[skin, 2] > vertices[:, 2].mean())
  ]
  if band.size == 0:
    raise ValueError('No cheek band found; the vertex groups have changed.')

  weight = np.zeros(len(vertices))
  for side in (band[vertices[band, 0] > 0], band[vertices[band, 0] < 0]):
    centre = vertices[side].mean(axis=0)
    distance = np.linalg.norm(vertices[skin] - centre, axis=1)
    # Squared to make it C1 at the rim, so the puff has no visible seam.
    falloff = np.clip(1.0 - (distance / radius) ** 2, 0.0, 1.0) ** 2
    weight[skin] = np.maximum(weight[skin], falloff)

  return (normals * (weight[:, None] * amplitude)).astype(np.float32)


def _vertex_normals(
    vertices: npt.NDArray[np.floating],
    triangles: npt.NDArray[np.integer],
) -> npt.NDArray[np.float64]:
  """Area-weighted vertex normals, unit length."""
  a, b, c = (vertices[triangles[:, k]] for k in range(3))
  face = np.cross(b - a, c - a)
  normals = np.zeros_like(vertices)
  for k in range(3):
    np.add.at(normals, triangles[:, k], face)
  lengths = np.linalg.norm(normals, axis=1, keepdims=True)
  return normals / np.maximum(lengths, 1e-12)


def _hsl_to_rgb(
    hue: npt.NDArray[np.floating],
    saturation: float,
    lightness: npt.NDArray[np.floating],
) -> npt.NDArray[np.uint8]:
  """Converts HSL to 8-bit RGB, vectorized over the inputs.

  Args:
    hue: Hue in [0, 1), (N,).
    saturation: Saturation in [0, 1], scalar.
    lightness: Lightness in [0, 1], (N,).

  Returns:
    RGB bytes, (N, 3).
  """
  chroma = (1.0 - np.abs(2.0 * lightness - 1.0)) * saturation
  sector = hue * 6.0
  second = chroma * (1.0 - np.abs(np.mod(sector, 2.0) - 1.0))
  base = lightness - chroma / 2.0

  zeros = np.zeros_like(hue)
  table = np.stack([
      np.stack([chroma, second, zeros], axis=-1),
      np.stack([second, chroma, zeros], axis=-1),
      np.stack([zeros, chroma, second], axis=-1),
      np.stack([zeros, second, chroma], axis=-1),
      np.stack([second, zeros, chroma], axis=-1),
      np.stack([chroma, zeros, second], axis=-1),
  ])
  rgb = table[np.clip(sector.astype(np.int32), 0, 5), np.arange(hue.size)]
  return np.clip((rgb + base[:, None]) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def correspondence_colors(
    positions: npt.NDArray[np.floating],
) -> npt.NDArray[np.uint8]:
  """Assigns each corresponding point a colour that identifies its location.

  The whole point of drawing these twice -- once on the webcam image, once on
  the mesh -- is being able to say "that point is that point". Index-derived
  colours would not do it, because neighbouring MediaPipe indices are not
  neighbouring positions. Deriving hue from horizontal position and lightness
  from vertical position instead makes the colour a readable function of where
  the point sits on a face, so the eye can match the two views directly.

  Args:
    positions: Template positions of the corresponding vertices, (N, 3).

  Returns:
    RGB bytes, (N, 3).
  """
  def normalize(values):
    low, high = values.min(), values.max()
    return (values - low) / max(high - low, 1e-9)

  # 0.82 rather than a full turn, so the two ends of the face do not both come
  # out red and become impossible to tell apart.
  hue = normalize(positions[:, 0]) * 0.82
  # Kept well below white: these are drawn over a bloomed render, and anything
  # bright enough to bloom loses its hue, which is the only information the
  # colour carries.
  lightness = 0.34 + normalize(positions[:, 1]) * 0.24
  return _hsl_to_rgb(hue, 0.95, lightness)


def load_correspondence(
    path: str,
    vertex_count: int,
) -> dict[str, npt.NDArray[Any]] | None:
  """Loads the landmark to vertex mapping, if it is present.

  Args:
    path: Path to the correspondence `.npz`.
    vertex_count: Number of mesh vertices, for validation.

  Returns:
    The landmark indices, vertex indices, rigid mask and neutral reference
    cloud, or None if the file does not exist.

  Raises:
    ValueError: If the file is present but does not match this model.
  """
  if not os.path.exists(path):
    logging.warning(
        'No correspondence at %s; the landmark overlay and identity fit will '
        'be unavailable. Build it with the webcam_puppet prototype.',
        path,
    )
    return None

  data = np.load(path, allow_pickle=True)
  landmark_indices = np.asarray(data['landmark_indices'], dtype=np.int32)
  vertex_indices = np.asarray(data['vertex_indices'], dtype=np.int32)
  if landmark_indices.shape != vertex_indices.shape:
    raise ValueError('Correspondence landmark and vertex arrays disagree.')
  if vertex_indices.max() >= vertex_count:
    raise ValueError(
        f'Correspondence references vertex {vertex_indices.max()} but the '
        f'model has {vertex_count}.'
    )

  # Where the landmarker puts these landmarks on the *neutral* head, in metric
  # model space. The identity fit needs this rather than the vertices they map
  # to: the landmarker reports its own idea of face shape, biased against GNM
  # by ~11 mm rms in z, and fitting displacement from this cloud cancels that
  # bias instead of absorbing it into identity.
  reference = np.asarray(data['reference_landmarks'], dtype=np.float32)
  if reference.shape != (len(landmark_indices), 3):
    raise ValueError(
        f'reference_landmarks must be {(len(landmark_indices), 3)}, got '
        f'{reference.shape}.'
    )

  return {
      'landmark_indices': landmark_indices.astype(np.uint16),
      'vertex_indices': vertex_indices.astype(np.uint16),
      'rigid': np.asarray(data['rigid']).astype(np.uint8),
      'reference': reference,
  }


def _as_uint16_bits(values: npt.NDArray[np.floating]) -> npt.NDArray[np.uint16]:
  """Reinterprets float16 values as the raw bits a half-float texture wants."""
  return values.astype(np.float16).view(np.uint16)


def build_arrays(
    gnm: gnm_numpy.GNM,
    identity_indices: npt.NDArray[np.integer],
    expression_indices: npt.NDArray[np.integer],
    correspondence: Mapping[str, npt.NDArray[Any]] | None = None,
) -> dict[str, npt.NDArray[Any]]:
  """Collects every array that goes into the binary blob.

  Args:
    gnm: The loaded model.
    identity_indices: Identity components to keep.
    expression_indices: Expression components to keep.
    correspondence: Optional landmark to vertex mapping.

  Returns:
    A mapping from array name to the array to serialize, in write order.
  """
  identity_basis = np.asarray(gnm.vertex_identity_basis)[identity_indices]
  expression_basis = np.asarray(gnm.expression_basis)[expression_indices]
  joint_identity_basis = np.asarray(gnm.joint_identity_basis)[identity_indices]

  triangles = np.asarray(gnm.triangles)
  if triangles.max() > np.iinfo(np.uint16).max:
    raise ValueError('Vertex count exceeds the uint16 index range.')

  # GNM head v3 ships an all-zero corrective regressor, so the whole stage is
  # a no-op: 3.9 MB of zeros to download and a matrix-exponential-per-joint
  # stage for the shader to run for nothing. It is dropped rather than
  # exported, and the guard fires if a later model actually populates it.
  if np.any(np.asarray(gnm.pose_correctives_regressor) != 0):
    raise ValueError(
        'This model has non-zero pose correctives, which the export drops. '
        'Re-add pose_correctives_regressor to the export and the corrective '
        'stage to reference_model and the vertex shader.'
    )

  adjacency_pairs, adjacency_counts, _ = normal_adjacency(
      triangles, int(gnm.num_vertices)
  )

  arrays: dict[str, npt.NDArray[Any]] = {
      'template_vertex_positions': np.asarray(
          gnm.template_vertex_positions, dtype=np.float32
      ),
      'identity_basis': _as_uint16_bits(identity_basis),
      'expression_basis': _as_uint16_bits(expression_basis),
      'template_joint_positions': np.asarray(
          gnm.template_joint_positions, dtype=np.float32
      ),
      'joint_identity_basis': joint_identity_basis.astype(np.float32),
      'skinning_weights': np.asarray(gnm.skinning_weights, dtype=np.float32),
      'triangles': triangles.astype(np.uint16),
      'quad_edges': quad_edges(np.asarray(gnm.quads)),
      'component_ids': component_ids(gnm),
      'normal_adjacency': adjacency_pairs,
      'normal_adjacency_count': adjacency_counts,
      # Shapes GNM cannot make, stacked like a tiny extra basis so the shader
      # applies them with the same loop it uses for expression.
      'corrective_basis': _as_uint16_bits(
          np.stack([cheek_puff_corrective(gnm)])
      ),
  }

  if correspondence is not None:
    template = np.asarray(gnm.template_vertex_positions, dtype=np.float32)
    arrays['correspondence_landmarks'] = correspondence['landmark_indices']
    arrays['correspondence_vertices'] = correspondence['vertex_indices']
    arrays['correspondence_rigid'] = correspondence['rigid']
    arrays['correspondence_reference'] = correspondence['reference']
    arrays['correspondence_colors'] = correspondence_colors(
        template[correspondence['vertex_indices'].astype(np.int32)]
    )
  return arrays


def serialize(
    arrays: Mapping[str, npt.NDArray[Any]],
) -> tuple[bytes, dict[str, Any]]:
  """Packs arrays into one blob and describes each as a view into it.

  Bases are stored as float16 bits but declared as `float16` in the manifest,
  because the consumer needs to know the semantic dtype, not the carrier.

  Args:
    arrays: Arrays to pack, in write order.

  Returns:
    A tuple of the packed blob and the per-array view descriptors.
  """
  # Names are the only place the half-float carrier is undone, so the manifest
  # reports what the numbers mean rather than how they travel.
  declared = {'identity_basis', 'expression_basis', 'corrective_basis'}

  blob = bytearray()
  views: dict[str, Any] = {}
  for name, array in arrays.items():
    padding = -len(blob) % _ALIGNMENT
    blob.extend(b'\0' * padding)

    contiguous = np.ascontiguousarray(array)
    payload = contiguous.tobytes()
    dtype = 'float16' if name in declared else _DTYPE_NAMES[contiguous.dtype]
    views[name] = {
        'dtype': dtype,
        'shape': list(contiguous.shape),
        'byteOffset': len(blob),
        'byteLength': len(payload),
    }
    blob.extend(payload)

  return bytes(blob), views


def export(
    output_dir: str,
    identity_components: int,
    expression_budget: Mapping[str, int],
) -> tuple[str, str]:
  """Writes the model assets and returns the two output paths.

  Args:
    output_dir: Directory to write into; created if missing.
    identity_components: Number of identity components to keep.
    expression_budget: Per-region expression component budget.

  Returns:
    A tuple of the binary path and the manifest path.
  """
  gnm = gnm_numpy.GNM.from_local(
      version=gnm_numpy.GNMMajorVersion.V3,
      variant=gnm_numpy.GNMVariant.HEAD,
  )

  expression_names = [str(name) for name in gnm.expression_names]
  identity_names = [str(name) for name in gnm.identity_names]

  identity_indices = np.arange(
      min(identity_components, len(identity_names)), dtype=np.int32
  )
  expression_indices = select_expression_components(
      expression_names, expression_budget
  )

  correspondence = load_correspondence(
      _CORRESPONDENCE_PATH.value, int(gnm.num_vertices)
  )
  arrays = build_arrays(
      gnm, identity_indices, expression_indices, correspondence
  )
  blob, views = serialize(arrays)

  kept_expression = [expression_names[i] for i in expression_indices]
  regions: dict[str, list[int]] = {}
  for position, name in enumerate(kept_expression):
    regions.setdefault(region_of(name), []).append(position)

  manifest = {
      'model': {
          'version': str(gnm.version),
          'variant': str(gnm.variant),
          'vertexCount': int(gnm.num_vertices),
          'jointCount': int(gnm.num_joints),
          'jointNames': [str(name) for name in gnm.joint_names],
          'jointParentIndices': [
              int(index) for index in gnm.joint_parent_indices
          ],
          'componentNames': [str(name) for name in gnm.mesh_component_names],
          # The GPU normal pass loops a compile-time constant number of
          # times, so it needs the padded width up front.
          'maxNormalCorners': views['normal_adjacency']['shape'][1],
      },
      'correspondence': {
          'count': 0 if correspondence is None
                   else len(correspondence['landmark_indices']),
      },
      # Not GNM. Authored shapes the model's basis cannot reach, each named
      # with the MediaPipe blendshape that drives it.
      'correctives': {
          'names': ['cheek_puff'],
          'blendshapes': ['cheekPuff'],
      },
      'identity': {
          'count': len(identity_indices),
          'sourceCount': len(identity_names),
          'indices': [int(i) for i in identity_indices],
          'names': [identity_names[i] for i in identity_indices],
      },
      'expression': {
          'count': len(expression_indices),
          'sourceCount': len(expression_names),
          'indices': [int(i) for i in expression_indices],
          'names': kept_expression,
          # Contiguous slice of the *kept* basis for each region, which is what
          # a per-region gain or ridge weight has to be applied over.
          'regions': {
              name: {'start': positions[0], 'count': len(positions)}
              for name, positions in regions.items()
          },
      },
      'arrays': views,
  }

  os.makedirs(output_dir, exist_ok=True)
  binary_path = os.path.join(output_dir, 'gnm_head.bin')
  manifest_path = os.path.join(output_dir, 'gnm_head.json')

  with open(binary_path, 'wb') as handle:
    handle.write(blob)
  with open(manifest_path, 'w', encoding='utf-8') as handle:
    json.dump(manifest, handle, indent=2)

  return binary_path, manifest_path


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError(f'Unexpected arguments: {argv[1:]}')

  binary_path, manifest_path = export(
      _OUTPUT_DIR.value,
      _IDENTITY_COMPONENTS.value,
      parse_budget(_EXPRESSION_BUDGET.value),
  )

  with open(manifest_path, encoding='utf-8') as handle:
    manifest = json.load(handle)

  logging.info('Wrote %s', manifest_path)
  logging.info(
      'Wrote %s (%.1f MB)',
      binary_path,
      os.path.getsize(binary_path) / 1e6,
  )
  header = ('array', 'dtype', 'shape', 'MB')
  print(f'{header[0]:<32}{header[1]:>9}{header[2]:>22}{header[3]:>8}')
  for name, view in manifest['arrays'].items():
    dtype = view['dtype']
    shape = str(tuple(view['shape']))
    megabytes = view['byteLength'] / 1e6
    print(f'{name:<32}{dtype:>9}{shape:>22}{megabytes:>8.2f}')
  total = os.path.getsize(binary_path) / 1e6
  label, blank = 'total', ''
  print(f'{label:<32}{blank:>9}{blank:>22}{total:>8.2f}')

  identity = manifest['identity']
  expression = manifest['expression']
  parts = []
  for name, region in expression['regions'].items():
    count = region['count']
    parts.append(f'{name} {count}')
  breakdown = ', '.join(parts)
  identity_kept = identity['count']
  identity_total = identity['sourceCount']
  expression_kept = expression['count']
  expression_total = expression['sourceCount']
  print(
      f'\nidentity {identity_kept}/{identity_total} components, '
      f'expression {expression_kept}/{expression_total} '
      f'components ({breakdown})'
  )


if __name__ == '__main__':
  app.run(main)
