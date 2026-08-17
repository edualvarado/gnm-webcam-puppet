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

"""Verifies the exported web assets reproduce `gnm.shape`.

Two error sources are checked separately, because only one of them is a bug:

  * **Fidelity** -- the reference model fed the kept coefficients must match
    `gnm.shape` fed the same coefficients scattered back into a full-length
    vector. Any disagreement here is a packing, layout or dtype error, and the
    tolerance is tight.
  * **Truncation** -- dropping components necessarily loses shape. That is a
    budget choice, not a defect, so it is measured and reported rather than
    asserted against a tight bound.

Run from the repository root:
  python -m web_puppet.tools.export_assets_test
"""

import tempfile

from absl.testing import absltest
from absl.testing import parameterized
from gnm.shape import gnm_numpy
import numpy as np

from web_puppet.tools import export_assets
from web_puppet.tools import reference_model

# Fidelity is asserted relative to how far the mesh actually deformed, not as
# an absolute distance: float16 error is proportional to the magnitude it
# encodes, so an absolute bound would pass or fail on the draw scale rather
# than on correctness. At float32 the reference is bit-exact with `gnm.shape`,
# so everything below this bound is quantization and nothing else. Measured
# worst case is ~3e-4; a packing, stride or dtype error is order 1.
_FIDELITY_RELATIVE_TOLERANCE = 1e-3

_TEST_BUDGET = {
    'left_eye_region': 24,
    'right_eye_region': 24,
    'lower_face_region': 32,
    'tongue': 8,
    'pupils': 1,
}
_TEST_IDENTITY_COMPONENTS = 64


def _to_mm(deltas: np.ndarray) -> np.ndarray:
  """Converts per-vertex displacement vectors in metres to millimetres."""
  return np.linalg.norm(deltas, axis=-1) * 1000.0


class ExportAssetsTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.asset_dir = tempfile.mkdtemp()
    export_assets.export(
        cls.asset_dir, _TEST_IDENTITY_COMPONENTS, _TEST_BUDGET
    )
    cls.reference = reference_model.ReferenceModel.load(cls.asset_dir)
    cls.gnm = gnm_numpy.GNM.from_local(
        version=gnm_numpy.GNMMajorVersion.V3,
        variant=gnm_numpy.GNMVariant.HEAD,
    )

  def _random_parameters(self, seed: int, scale: float = 1.0):
    """Draws kept-space identity, expression, rotations and translation."""
    rng = np.random.default_rng(seed)
    identity = rng.normal(
        0.0, scale, self.reference.identity_dim
    ).astype(np.float32)
    expression = rng.normal(
        0.0, scale, self.reference.expression_dim
    ).astype(np.float32)
    rotations = rng.normal(
        0.0, 0.15, (self.reference.num_joints, 3)
    ).astype(np.float32)
    translation = rng.normal(0.0, 0.02, 3).astype(np.float32)
    return identity, expression, rotations, translation

  def test_topology_is_well_formed(self):
    reference = self.reference
    self.assertEqual(reference.num_vertices, self.gnm.num_vertices)
    self.assertEqual(reference.num_joints, self.gnm.num_joints)

    self.assertLess(reference.triangles.max(), reference.num_vertices)
    self.assertGreaterEqual(reference.triangles.min(), 0)
    np.testing.assert_array_equal(
        reference.triangles, np.asarray(self.gnm.triangles)
    )

    # Every quad edge must be a real pair of distinct vertices, and the list
    # must be deduplicated -- the wireframe overdraws otherwise.
    self.assertLess(reference.quad_edges.max(), reference.num_vertices)
    self.assertTrue(
        np.all(reference.quad_edges[:, 0] != reference.quad_edges[:, 1])
    )
    self.assertLen(
        np.unique(reference.quad_edges, axis=0), len(reference.quad_edges)
    )

  def test_component_ids_partition_the_mesh(self):
    names = self.reference.manifest['model']['componentNames']
    self.assertEqual(names, list(self.gnm.mesh_component_names))

    ids = self.reference.component_ids
    self.assertLen(ids, self.reference.num_vertices)
    self.assertGreaterEqual(ids.min(), 0)
    self.assertLess(ids.max(), len(names))

    # Each vertex must land in the component whose group actually claims it.
    for index, name in enumerate(names):
      group = np.asarray(self.gnm.vertex_group(name)) > 0
      np.testing.assert_array_equal(ids[group], index)

  def test_expression_regions_cover_the_kept_basis(self):
    expression = self.reference.manifest['expression']
    covered = sorted(
        position
        for region in expression['regions'].values()
        for position in range(
            region['start'], region['start'] + region['count']
        )
    )
    self.assertEqual(covered, list(range(expression['count'])))

    for name, region in expression['regions'].items():
      kept = expression['names'][
          region['start'] : region['start'] + region['count']
      ]
      self.assertEqual({export_assets.region_of(n) for n in kept}, {name})
      self.assertEqual(region['count'], _TEST_BUDGET[name])

  @parameterized.named_parameters(
      ('neutral', 0, 0.0),
      ('mild', 1, 0.5),
      ('strong', 2, 1.5),
      ('extreme', 3, 3.0),
  )
  def test_matches_gnm_on_kept_components(self, seed, scale):
    identity, expression, rotations, translation = self._random_parameters(
        seed, scale
    )

    actual = self.reference(identity, expression, rotations, translation)
    expected = self.gnm(
        identity=self.reference.expand_identity(identity),
        expression=self.reference.expand_expression(expression),
        rotations=rotations,
        translation=translation,
    )

    error = _to_mm(np.asarray(actual) - np.asarray(expected))
    # Scale the bound by the deformation the parameters actually produced, so
    # a neutral draw is held to the same standard as an extreme one.
    extent = _to_mm(
        np.asarray(expected) - self.reference.template_vertex_positions
    ).max()
    tolerance = _FIDELITY_RELATIVE_TOLERANCE * max(extent, 1.0)
    self.assertLess(
        error.max(),
        tolerance,
        f'max {error.max():.6f} mm, rms {np.sqrt((error**2).mean()):.6f} mm, '
        f'deformation extent {extent:.2f} mm',
    )

  def test_pose_correctives_are_a_no_op(self):
    # The export drops the corrective stage because this model's regressor is
    # entirely zero. If that ever stops being true the export must grow the
    # stage back, so pin the fact here rather than leaving it to a comment.
    regressor = np.asarray(self.gnm.pose_correctives_regressor)
    self.assertTrue(
        np.all(regressor == 0),
        'Model now ships non-zero pose correctives; the exported assets and '
        'the WebGL forward pass must implement the corrective stage.',
    )
    self.assertNotIn('pose_correctives_regressor', self.reference.manifest[
        'arrays'
    ])

  def test_normal_adjacency_reproduces_face_normals(self):
    # The GPU rebuilds normals from this adjacency instead of iterating
    # triangles, so it has to give the identical answer -- including
    # magnitudes, which carry the area weighting. Anything approximate here
    # would show up as shading that drifts from the CPU reference.
    reference = self.reference
    arrays = reference_model._read_views(  # pylint: disable=protected-access
        open(  # pylint: disable=consider-using-with
            f'{self.asset_dir}/gnm_head.bin', 'rb'
        ).read(),
        reference.manifest['arrays'],
    )
    pairs = arrays['normal_adjacency'].astype(np.int32)
    counts = arrays['normal_adjacency_count'].astype(np.int32)

    rng = np.random.default_rng(11)
    identity = rng.normal(0.0, 1.0, reference.identity_dim).astype(np.float32)
    expression = rng.normal(
        0.0, 1.0, reference.expression_dim
    ).astype(np.float32)
    vertices = reference.bind_pose_vertices(identity, expression)

    # The straightforward triangle-iterating accumulation, i.e. what the
    # TypeScript CPU path does.
    expected = np.zeros_like(vertices)
    corner_a = vertices[reference.triangles[:, 0]]
    corner_b = vertices[reference.triangles[:, 1]]
    corner_c = vertices[reference.triangles[:, 2]]
    face = np.cross(corner_b - corner_a, corner_c - corner_a)
    for corner in range(3):
      np.add.at(expected, reference.triangles[:, corner], face)

    # The adjacency-driven accumulation, i.e. what the GPU pass does.
    actual = np.zeros_like(vertices)
    for slot in range(pairs.shape[1]):
      active = counts > slot
      if not np.any(active):
        break
      origin = vertices[active]
      first = vertices[pairs[active, slot, 0]]
      second = vertices[pairs[active, slot, 1]]
      actual[active] += np.cross(first - origin, second - origin)

    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-9)
    self.assertGreaterEqual(counts.min(), 1)

  def test_correspondence_is_consistent(self):
    manifest = self.reference.manifest
    if manifest['correspondence']['count'] == 0:
      self.skipTest('No correspondence.npz available to export.')

    arrays = reference_model._read_views(  # pylint: disable=protected-access
        open(  # pylint: disable=consider-using-with
            f'{self.asset_dir}/gnm_head.bin', 'rb'
        ).read(),
        manifest['arrays'],
    )
    landmarks = arrays['correspondence_landmarks'].astype(np.int32)
    vertices = arrays['correspondence_vertices'].astype(np.int32)
    colors = arrays['correspondence_colors']

    count = manifest['correspondence']['count']
    self.assertLen(landmarks, count)
    self.assertLen(vertices, count)
    self.assertEqual(colors.shape, (count, 3))

    # MediaPipe's face mesh has 478 landmarks; anything outside that would
    # index off the end of a detection at runtime.
    self.assertLess(landmarks.max(), 478)
    self.assertLess(vertices.max(), self.reference.num_vertices)
    self.assertLen(np.unique(landmarks), count)

    # Every corresponding vertex must be skin. A mapping that landed on an
    # eyeball or a tooth would draw a marker inside the head.
    skin = np.asarray(self.gnm.vertex_group('skin')) > 0
    self.assertTrue(np.all(skin[vertices]))

    # Colour has to actually vary across the face, since telling two points
    # apart by eye is the only thing it is for.
    self.assertGreater(int(colors.max()) - int(colors.min()), 100)

  def test_reports_truncation_error(self):
    # Not a pass/fail bound -- the budget is a size/quality tradeoff. This
    # prints the number that justifies it, over parameters drawn across the
    # full model rather than only the components that were kept.
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(8):
      identity = rng.normal(0.0, 1.0, self.gnm.identity_dim).astype(np.float32)
      expression = rng.normal(
          0.0, 1.0, self.gnm.expression_dim
      ).astype(np.float32)

      full = np.asarray(self.gnm(identity=identity, expression=expression))
      kept_identity = identity[
          np.asarray(self.reference.manifest['identity']['indices'])
      ]
      kept_expression = expression[
          np.asarray(self.reference.manifest['expression']['indices'])
      ]
      truncated = self.reference(kept_identity, kept_expression)

      error = _to_mm(full - truncated)
      worst = max(worst, error.max())
      print(
          f'truncation: max {error.max():6.3f} mm  '
          f'rms {np.sqrt((error ** 2).mean()):6.3f} mm'
      )

    # Loose sanity bound: an unposed head is ~200 mm tall, so anything past a
    # centimetre means the budget dropped something structural.
    self.assertLess(worst, 10.0)


if __name__ == '__main__':
  absltest.main()
