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

"""Real-time triangle rasterizer for GNM meshes.

The GNM head has ~35k triangles, which cannot be drawn one at a time from
Python at interactive rates. This renderer keeps the whole frame vectorized in
NumPy by expanding each triangle into its scanline spans in one pass: every
triangle contributes exactly the pixels it covers, with no per-triangle Python
work and no wasted samples outside the shape.

Shading is Gouraud, carried as a scalar light intensity rather than RGB so that
interpolation touches a third of the memory, with albedo and gamma applied
through a 256-entry lookup table at the very end.
"""

from collections.abc import Sequence
import dataclasses

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating]
IntArray = npt.NDArray[np.integer]
ByteArray = npt.NDArray[np.uint8]

# Direction the scene light travels *from*, in camera space (x right, y up,
# z towards the viewer). Slightly up and to the left reads as conventional
# portrait lighting.
_DEFAULT_LIGHT_DIRECTION = (-0.3, 0.4, 1.0)
_AMBIENT = 0.25
_DEFAULT_SKIN_COLOR = (0.86, 0.72, 0.64)
_DEFAULT_BACKGROUND = (0.09, 0.10, 0.12)

# Sign of the screen-space signed area of a front-facing triangle. Projection
# flips Y, so a triangle whose outward normal -- cross(b - a, c - a), the same
# winding compute_vertex_normals assumes -- points at the camera comes out
# clockwise on screen, which is negative.
_FRONT_FACING_SIGN = np.float32(-1.0)

# Slack in pixels when turning a span's real endpoints into integer columns.
# Without it, rounding at a shared edge can leave the pixel unclaimed by both
# triangles, which shows up as isolated dark specks along creases.
_SPAN_TOLERANCE = np.float32(1e-3)

_SHADE_LEVELS = 256


def compute_vertex_normals(
    vertices: FloatArray,
    triangles: IntArray,
) -> FloatArray:
  """Computes unit vertex normals by area-weighted face-normal accumulation.

  Args:
    vertices: Vertex positions, (V, 3).
    triangles: Triangle vertex indices, (T, 3).

  Returns:
    Unit-length per-vertex normals, (V, 3).
  """
  corner_a = vertices[triangles[:, 0]]
  corner_b = vertices[triangles[:, 1]]
  corner_c = vertices[triangles[:, 2]]

  # The cross product magnitude is twice the triangle area, so accumulating it
  # un-normalized area-weights each face's contribution for free.
  face_normals = np.cross(corner_b - corner_a, corner_c - corner_a)

  normals = np.zeros_like(vertices)
  for corner in range(3):
    np.add.at(normals, triangles[:, corner], face_normals)

  lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
  return normals / np.maximum(lengths, 1e-12)


def _encode(linear: FloatArray) -> ByteArray:
  """Gamma-encodes linear RGB in [0, 1] to display-referred bytes."""
  return (np.clip(linear, 0.0, 1.0) ** (1.0 / 2.2) * 255.0 + 0.5).astype(
      np.uint8
  )


def _expand(counts: IntArray) -> tuple[IntArray, IntArray]:
  """Flattens per-group counts into (group index, position within group).

  Args:
    counts: Non-negative item count for each group, (G,).

  Returns:
    A tuple of the owning group index and the 0-based position within that
    group, both (sum(counts),).
  """
  group = np.repeat(np.arange(counts.size, dtype=np.int32), counts)
  starts = np.cumsum(counts, dtype=np.int32) - counts
  position = np.arange(group.size, dtype=np.int32) - starts[group]
  return group, position


@dataclasses.dataclass(frozen=True)
class Camera:
  """A pinhole camera looking down -Z at a fixed target.

  Attributes:
    image_size: Output resolution as (height, width).
    target: The world-space point the camera looks at, (3,).
    distance: Distance from the target along +Z.
    focal_length: Focal length in pixels.
  """

  image_size: tuple[int, int]
  target: FloatArray
  distance: float
  focal_length: float

  @classmethod
  def fit_to_mesh(
      cls,
      vertices: FloatArray,
      image_size: tuple[int, int] = (480, 480),
      fill_factor: float = 0.85,
  ) -> 'Camera':
    """Builds a camera framing the mesh's bounding box.

    Args:
      vertices: Vertex positions used to measure the subject, (V, 3).
      image_size: Output resolution as (height, width).
      fill_factor: Fraction of the smaller image axis the subject should span.

    Returns:
      A camera that frames the mesh.
    """
    lower = vertices.min(axis=0)
    upper = vertices.max(axis=0)
    target = (lower + upper) / 2.0

    # Frame the taller of the two in-plane extents so the subject always fits.
    subject_extent = float(np.max(upper[:2] - lower[:2]))
    depth_extent = float(upper[2] - lower[2])
    distance = 2.5 * subject_extent + depth_extent

    focal_length = fill_factor * min(image_size) * distance / subject_extent
    return cls(
        image_size=image_size,
        target=np.asarray(target, dtype=np.float32),
        distance=float(distance),
        focal_length=float(focal_length),
    )

  def project(self, vertices: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Projects world-space points to pixel coordinates.

    Args:
      vertices: World-space positions, (V, 3).

    Returns:
      A tuple of pixel coordinates as float (V, 2) in (x, y) order, and camera
      -space depth (V,), increasing away from the camera.
    """
    height, width = self.image_size
    centered = vertices - self.target

    # Camera sits at +Z looking towards -Z, so depth grows as z decreases.
    depth = self.distance - centered[..., 2]
    safe_depth = np.maximum(depth, 1e-6)

    scale = self.focal_length / safe_depth
    pixel_x = centered[..., 0] * scale + width / 2.0
    # Image rows grow downward while world Y grows upward.
    pixel_y = -centered[..., 1] * scale + height / 2.0
    return np.stack([pixel_x, pixel_y], axis=-1), depth


class MeshRenderer:
  """Rasterizes a mesh as a smooth-shaded surface with an exact z-buffer."""

  def __init__(
      self,
      triangles: IntArray,
      camera: Camera,
      background: Sequence[float] = _DEFAULT_BACKGROUND,
      light_direction: Sequence[float] = _DEFAULT_LIGHT_DIRECTION,
      base_color: Sequence[float] = _DEFAULT_SKIN_COLOR,
  ):
    """Builds the renderer.

    Args:
      triangles: Triangle vertex indices, (T, 3).
      camera: The camera to render with.
      background: Background color as linear RGB in [0, 1].
      light_direction: Direction the light arrives from, in camera space.
      base_color: Diffuse albedo as linear RGB in [0, 1].
    """
    self._triangles = np.ascontiguousarray(triangles, dtype=np.int32)
    self._camera = camera
    self._background = _encode(np.asarray(background, dtype=np.float32))

    light = np.asarray(light_direction, dtype=np.float32)
    self._light_direction = light / np.linalg.norm(light)

    # Shading only ever scales one albedo, so resolve albedo and gamma once
    # into a ramp and let each pixel index into it.
    ramp = np.linspace(0.0, 1.0, _SHADE_LEVELS, dtype=np.float32)
    self._shade_table = _encode(
        ramp[:, None] * np.asarray(base_color, dtype=np.float32)
    )

  @property
  def camera(self) -> Camera:
    """The camera this renderer draws with."""
    return self._camera

  def light_intensity(self, normals: FloatArray) -> FloatArray:
    """Computes per-vertex Lambertian intensity in [0, 1].

    Args:
      normals: Unit vertex normals, (V, 3).

    Returns:
      Per-vertex light intensity, (V,).
    """
    diffuse = np.clip(normals @ self._light_direction, 0.0, 1.0)
    return _AMBIENT + (1.0 - _AMBIENT) * diffuse

  def render(self, vertices: FloatArray) -> ByteArray:
    """Renders the mesh to an RGB image.

    Args:
      vertices: Vertex positions, (V, 3).

    Returns:
      An RGB image of shape (height, width, 3), dtype uint8.
    """
    height, width = self._camera.image_size
    image = np.empty((height * width, 3), dtype=np.uint8)
    image[:] = self._background

    normals = compute_vertex_normals(vertices, self._triangles)
    levels = (
        self.light_intensity(normals) * (_SHADE_LEVELS - 1)
    ).astype(np.float32)

    pixels, depth = self._camera.project(vertices)
    pixel_x = np.ascontiguousarray(pixels[..., 0], dtype=np.float32)
    pixel_y = np.ascontiguousarray(pixels[..., 1], dtype=np.float32)
    # Depth is resolved as 1/z, which unlike z interpolates linearly across a
    # triangle in screen space, so visibility needs no per-pixel division.
    inv_depth = np.float32(1.0) / np.maximum(depth, 1e-6).astype(np.float32)

    corner_x = [pixel_x[self._triangles[:, k]] for k in range(3)]
    corner_y = [pixel_y[self._triangles[:, k]] for k in range(3)]

    area = (
        (corner_x[1] - corner_x[0]) * (corner_y[2] - corner_y[0])
        - (corner_y[1] - corner_y[0]) * (corner_x[2] - corner_x[0])
    ) * _FRONT_FACING_SIGN

    left = np.maximum(np.ceil(np.minimum.reduce(corner_x)), 0.0)
    top = np.maximum(np.ceil(np.minimum.reduce(corner_y)), 0.0)
    right = np.minimum(np.floor(np.maximum.reduce(corner_x)), width - 1)
    bottom = np.minimum(np.floor(np.maximum.reduce(corner_y)), height - 1)

    # A back-facing triangle has negative area after the sign flip, which also
    # discards the degenerate ones that project to a line.
    visible = np.flatnonzero((area > 1e-9) & (left <= right) & (top <= bottom))
    if visible.size == 0:
      return image.reshape((height, width, 3))

    origin_x = left[visible]
    origin_y = top[visible]
    columns = (right[visible] - origin_x + 1).astype(np.int32)
    rows = (bottom[visible] - origin_y + 1).astype(np.int32)
    inv_area = np.float32(1.0) / area[visible]

    # Edge functions are evaluated in bounding-box-local pixel coordinates. In
    # absolute coordinates their constant term is a product of two ~500 px
    # values while the result is order 1, and cancellation at that ratio opens
    # single-pixel cracks along shared edges in float32.
    local_x = [corner_x[k][visible] - origin_x for k in range(3)]
    local_y = [corner_y[k][visible] - origin_y for k in range(3)]

    # edge[k](x, y) = a*x + b*y + c, positive inside, one per triangle side.
    edge_a, edge_b, edge_c = [], [], []
    for tail, head in ((0, 1), (1, 2), (2, 0)):
      edge_a.append(_FRONT_FACING_SIGN * (local_y[tail] - local_y[head]))
      edge_b.append(_FRONT_FACING_SIGN * (local_x[head] - local_x[tail]))
      edge_c.append(
          _FRONT_FACING_SIGN
          * (local_x[tail] * local_y[head] - local_y[tail] * local_x[head])
      )

    span_triangle, span_row = _expand(rows)
    row = span_row.astype(np.float32)

    # Each edge bounds the covered columns of a scanline from one side: solving
    # a*x >= -(b*y + c) gives a lower bound where a is positive and an upper
    # bound where it is negative. A horizontal edge (a == 0) constrains nothing
    # and instead rejects the whole scanline when it falls outside.
    first = np.zeros(span_row.size, dtype=np.float32)
    last = (columns[span_triangle] - 1).astype(np.float32)
    for k in range(3):
      slope = edge_a[k][span_triangle]
      offset = -(
          edge_b[k][span_triangle] * row + edge_c[k][span_triangle]
      )
      with np.errstate(divide='ignore', invalid='ignore'):
        crossing = offset / slope
      first = np.where(slope > 0.0, np.maximum(first, crossing), first)
      last = np.where(slope < 0.0, np.minimum(last, crossing), last)
      last = np.where((slope == 0.0) & (offset > 0.0), first - 1.0, last)

    start = np.ceil(first - _SPAN_TOLERANCE).astype(np.int32)
    stop = np.floor(last + _SPAN_TOLERANCE).astype(np.int32)
    covered = np.maximum(stop - start + 1, 0)

    span_of, step = _expand(covered)
    if span_of.size == 0:
      return image.reshape((height, width, 3))

    triangle = span_triangle[span_of]
    x = (start[span_of] + step).astype(np.float32)
    y = row[span_of]
    pixel_ids = (
        (origin_y[triangle] + y) * width + origin_x[triangle] + x
    ).astype(np.int32)

    scale = inv_area[triangle]
    weight_a = (
        edge_a[1][triangle] * x + edge_b[1][triangle] * y + edge_c[1][triangle]
    ) * scale
    weight_b = (
        edge_a[2][triangle] * x + edge_b[2][triangle] * y + edge_c[2][triangle]
    ) * scale
    # The three edge functions sum to the signed area, so the last weight is
    # whatever the other two leave behind.
    weight_c = np.float32(1.0) - weight_a - weight_b

    corners = self._triangles[visible][triangle]
    fragment_inv_depth = (
        weight_a * inv_depth[corners[:, 0]]
        + weight_b * inv_depth[corners[:, 1]]
        + weight_c * inv_depth[corners[:, 2]]
    )

    # Scatter-max keeps the nearest fragment per pixel, then the survivors are
    # the ones whose own value came back out of the buffer. The comparison is
    # exact equality, so the buffer must carry the same dtype the fragments do
    # or it rounds them on write and matches almost nothing.
    buffer = np.zeros(height * width, dtype=fragment_inv_depth.dtype)
    np.maximum.at(buffer, pixel_ids, fragment_inv_depth)
    winners = np.flatnonzero(fragment_inv_depth == buffer[pixel_ids])

    corners = corners[winners]
    intensity = (
        weight_a[winners] * levels[corners[:, 0]]
        + weight_b[winners] * levels[corners[:, 1]]
        + weight_c[winners] * levels[corners[:, 2]]
    )
    level = np.clip(intensity, 0, _SHADE_LEVELS - 1).astype(np.uint8)
    image[pixel_ids[winners]] = self._shade_table[level]
    return image.reshape((height, width, 3))
