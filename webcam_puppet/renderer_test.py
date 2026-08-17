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

"""Tests for the mesh renderer."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from webcam_puppet import renderer as renderer_lib


def _unit_quad() -> tuple[np.ndarray, np.ndarray]:
  """Returns a unit quad in the z=0 plane facing +Z, as two triangles."""
  vertices = np.array(
      [
          [-0.5, -0.5, 0.0],
          [0.5, -0.5, 0.0],
          [0.5, 0.5, 0.0],
          [-0.5, 0.5, 0.0],
      ],
      dtype=np.float32,
  )
  triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
  return vertices, triangles


class ComputeVertexNormalsTest(parameterized.TestCase):

  def test_planar_quad_normals_face_positive_z(self):
    vertices, triangles = _unit_quad()

    normals = renderer_lib.compute_vertex_normals(vertices, triangles)

    np.testing.assert_allclose(
        normals, np.tile([0.0, 0.0, 1.0], (4, 1)), atol=1e-6
    )

  def test_normals_are_unit_length(self):
    rng = np.random.default_rng(0)
    vertices = rng.normal(size=(50, 3)).astype(np.float32)
    triangles = rng.integers(0, 50, size=(80, 3)).astype(np.int32)

    normals = renderer_lib.compute_vertex_normals(vertices, triangles)

    lengths = np.linalg.norm(normals, axis=-1)
    # Vertices touched by no triangle stay at zero rather than becoming NaN.
    self.assertTrue(np.all((np.abs(lengths - 1.0) < 1e-5) | (lengths == 0.0)))

  def test_isolated_vertex_does_not_produce_nan(self):
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [5.0, 5.0, 5.0]],
        dtype=np.float32,
    )
    triangles = np.array([[0, 1, 2]], dtype=np.int32)

    normals = renderer_lib.compute_vertex_normals(vertices, triangles)

    self.assertFalse(np.any(np.isnan(normals)))


class CameraTest(parameterized.TestCase):

  def test_target_projects_to_image_center(self):
    camera = renderer_lib.Camera(
        image_size=(200, 300),
        target=np.zeros(3, dtype=np.float32),
        distance=1.0,
        focal_length=100.0,
    )

    pixels, depth = camera.project(np.zeros((1, 3), dtype=np.float32))

    np.testing.assert_allclose(pixels[0], [150.0, 100.0], atol=1e-5)
    np.testing.assert_allclose(depth[0], 1.0, atol=1e-6)

  def test_world_up_maps_to_smaller_row(self):
    camera = renderer_lib.Camera(
        image_size=(200, 200),
        target=np.zeros(3, dtype=np.float32),
        distance=1.0,
        focal_length=100.0,
    )

    points = np.array([[0.0, 0.1, 0.0], [0.0, -0.1, 0.0]], dtype=np.float32)
    pixels, _ = camera.project(points)

    self.assertLess(pixels[0, 1], pixels[1, 1])

  def test_closer_point_has_smaller_depth(self):
    camera = renderer_lib.Camera(
        image_size=(100, 100),
        target=np.zeros(3, dtype=np.float32),
        distance=2.0,
        focal_length=100.0,
    )

    points = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, -0.5]], dtype=np.float32)
    _, depth = camera.project(points)

    self.assertLess(depth[0], depth[1])

  def test_fit_to_mesh_frames_subject_within_image(self):
    vertices, _ = _unit_quad()

    camera = renderer_lib.Camera.fit_to_mesh(vertices, image_size=(256, 256))
    pixels, _ = camera.project(vertices)

    self.assertTrue(np.all(pixels >= 0.0))
    self.assertTrue(np.all(pixels <= 256.0))


def _facing_camera_quad(depth: float) -> tuple[np.ndarray, np.ndarray]:
  """Returns a quad at the given z, wound so its normal points at the camera."""
  vertices, triangles = _unit_quad()
  vertices = vertices.copy()
  vertices[:, 2] = depth
  return vertices, triangles


class MeshRendererTest(parameterized.TestCase):

  def test_render_returns_expected_shape_and_dtype(self):
    vertices, triangles = _unit_quad()
    camera = renderer_lib.Camera.fit_to_mesh(vertices, image_size=(64, 96))
    renderer = renderer_lib.MeshRenderer(triangles, camera)

    image = renderer.render(vertices)

    self.assertEqual(image.shape, (64, 96, 3))
    self.assertEqual(image.dtype, np.uint8)

  def test_offscreen_geometry_leaves_background(self):
    vertices, triangles = _unit_quad()
    camera = renderer_lib.Camera(
        image_size=(32, 32),
        target=np.array([1000.0, 1000.0, 0.0], dtype=np.float32),
        distance=1.0,
        focal_length=100.0,
    )
    renderer = renderer_lib.MeshRenderer(
        triangles, camera, background=(0.0, 0.0, 0.0)
    )

    image = renderer.render(vertices)

    np.testing.assert_array_equal(image, np.zeros_like(image))

  def test_triangle_interior_is_filled(self):
    # The defining difference from splatting: coverage comes from the faces,
    # so a two-triangle quad leaves no gaps between its four vertices.
    vertices, triangles = _unit_quad()
    camera = renderer_lib.Camera.fit_to_mesh(vertices, image_size=(64, 64))
    renderer = renderer_lib.MeshRenderer(
        triangles, camera, background=(0.0, 0.0, 0.0)
    )

    image = renderer.render(vertices)

    covered = np.any(image > 0, axis=-1)
    rows, columns = np.nonzero(covered)
    interior = covered[
        rows.min():rows.max() + 1, columns.min():columns.max() + 1
    ]
    self.assertTrue(np.all(interior))

  def test_back_facing_geometry_is_culled(self):
    vertices, triangles = _unit_quad()
    camera = renderer_lib.Camera.fit_to_mesh(vertices, image_size=(64, 64))
    renderer = renderer_lib.MeshRenderer(
        triangles, camera, background=(0.0, 0.0, 0.0)
    )

    facing = renderer.render(vertices)
    self.assertTrue(np.any(facing > 0))

    # Reversing the winding turns the same geometry away from the camera.
    turned_away = renderer_lib.MeshRenderer(
        triangles[:, ::-1].copy(), camera, background=(0.0, 0.0, 0.0)
    )
    np.testing.assert_array_equal(
        turned_away.render(vertices), np.zeros_like(facing)
    )

  def test_nearer_surface_occludes_farther_one(self):
    near_vertices, triangles = _facing_camera_quad(0.3)
    far_vertices, _ = _facing_camera_quad(-0.3)
    vertices = np.concatenate([near_vertices, far_vertices])
    both = np.concatenate([triangles, triangles + len(near_vertices)])

    camera = renderer_lib.Camera(
        image_size=(32, 32),
        target=np.zeros(3, dtype=np.float32),
        distance=2.0,
        focal_length=40.0,
    )
    # A brighter light on the near quad would be indistinguishable, so instead
    # compare against rendering each quad alone.
    renderer = renderer_lib.MeshRenderer(both, camera)
    near_only = renderer_lib.MeshRenderer(triangles, camera)

    combined = renderer.render(vertices)
    expected = near_only.render(near_vertices)

    np.testing.assert_array_equal(combined[16, 16], expected[16, 16])

  @parameterized.named_parameters(
      ('float32', np.float32),
      ('float64', np.float64),
  )
  def test_coverage_does_not_depend_on_vertex_dtype(self, dtype):
    # Regression guard: the z-buffer identifies visible fragments by exact
    # depth equality, so a buffer narrower than the incoming depths rounds them
    # on write and discards nearly everything. GNM's forward pass returns
    # float64 even though its template vertices are float32.
    vertices, triangles = _unit_quad()
    camera = renderer_lib.Camera.fit_to_mesh(vertices, image_size=(128, 128))
    renderer = renderer_lib.MeshRenderer(
        triangles, camera, background=(0.0, 0.0, 0.0)
    )

    image = renderer.render(vertices.astype(dtype))

    covered = int(np.count_nonzero(np.any(image > 0, axis=-1)))
    self.assertGreater(covered, 5000)

  def test_render_is_invariant_to_triangle_ordering(self):
    # Guards the z-buffer against depending on input order.
    near_vertices, triangles = _facing_camera_quad(0.3)
    far_vertices, _ = _facing_camera_quad(-0.3)
    vertices = np.concatenate([near_vertices, far_vertices])
    both = np.concatenate([triangles, triangles + len(near_vertices)])

    camera = renderer_lib.Camera(
        image_size=(32, 32),
        target=np.zeros(3, dtype=np.float32),
        distance=2.0,
        focal_length=40.0,
    )

    forward = renderer_lib.MeshRenderer(both, camera).render(vertices)
    reversed_order = renderer_lib.MeshRenderer(both[::-1], camera).render(
        vertices
    )

    np.testing.assert_array_equal(forward, reversed_order)

  def test_light_intensity_is_brighter_facing_the_light(self):
    _, triangles = _unit_quad()
    camera = renderer_lib.Camera(
        image_size=(16, 16),
        target=np.zeros(3, dtype=np.float32),
        distance=1.0,
        focal_length=10.0,
    )
    renderer = renderer_lib.MeshRenderer(
        triangles, camera, light_direction=(0.0, 0.0, 1.0)
    )

    normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], dtype=np.float32)
    intensity = renderer.light_intensity(normals)

    self.assertGreater(float(intensity[0]), float(intensity[1]))

  def test_empty_mesh_renders_background(self):
    vertices, _ = _unit_quad()
    camera = renderer_lib.Camera.fit_to_mesh(vertices, image_size=(16, 16))
    renderer = renderer_lib.MeshRenderer(
        np.zeros((0, 3), dtype=np.int32), camera, background=(0.0, 0.0, 0.0)
    )

    image = renderer.render(vertices)

    np.testing.assert_array_equal(image, np.zeros_like(image))


if __name__ == '__main__':
  absltest.main()
