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

"""Pure-NumPy GNM forward pass over the *exported* web assets.

This deliberately does not import `gnm.shape`. It reads only the files the
browser will read, so it is a faithful executable spec of what the WebGL
implementation has to reproduce: if the shader and this agree, and this and
`gnm.shape` agree, the shader is correct.

Keeping it dependency-free is also what makes it a usable oracle -- the WebGL
side can be diffed against a stored dump from here without a Python runtime
anywhere near the browser.

The forward pass is, in order:
  1. bind-pose vertices = template + identity @ V_basis + expression @ E_basis
  2. bind-pose joints    = template_joints + identity @ J_basis
  3. linear blend skinning over the joint hierarchy

`gnm.shape` has a fourth stage between 2 and 3 -- pose correctives, which
regress vertex offsets from joint rotation. GNM head v3 ships an all-zero
regressor, so that stage contributes exactly nothing and is omitted here and
in the export. `export_assets` raises if a future model populates it.
"""

from collections.abc import Mapping
import dataclasses
import json
import os
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating]
IntArray = npt.NDArray[np.integer]

_NUMPY_DTYPES = {
    'float32': np.float32,
    'float16': np.float16,
    'uint16': np.uint16,
    'uint8': np.uint8,
}


def _read_views(
    blob: bytes,
    views: Mapping[str, Any],
) -> dict[str, npt.NDArray[Any]]:
  """Materializes each manifest view as an array over the blob.

  Args:
    blob: The packed binary payload.
    views: Per-array descriptors from the manifest.

  Returns:
    A mapping from array name to the array it describes.
  """
  arrays = {}
  for name, view in views.items():
    dtype = np.dtype(_NUMPY_DTYPES[view['dtype']])
    start = view['byteOffset']
    array = np.frombuffer(
        blob, dtype=dtype, count=view['byteLength'] // dtype.itemsize,
        offset=start,
    )
    arrays[name] = array.reshape(view['shape'])
  return arrays


def axis_angle_to_rotation_matrix(rotations: FloatArray) -> FloatArray:
  """Converts axis-angle vectors to rotation matrices via Rodrigues.

  Args:
    rotations: Axis-angle vectors, (..., 3).

  Returns:
    Rotation matrices, (..., 3, 3).
  """
  angle = np.linalg.norm(rotations, axis=-1, keepdims=True)
  # At zero angle the axis is undefined; any unit vector gives identity once
  # sin(0) and (1 - cos(0)) zero out both non-identity terms.
  axis = rotations / np.maximum(angle, 1e-12)

  x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
  zero = np.zeros_like(x)
  skew = np.stack(
      [zero, -z, y, z, zero, -x, -y, x, zero], axis=-1
  ).reshape(*x.shape, 3, 3)

  sin = np.sin(angle)[..., None]
  cos = np.cos(angle)[..., None]
  identity = np.eye(3, dtype=rotations.dtype)
  return identity + sin * skew + (1.0 - cos) * (skew @ skew)


@dataclasses.dataclass(frozen=True)
class ReferenceModel:
  """The exported GNM head, evaluated in NumPy.

  Attributes:
    manifest: The parsed JSON manifest.
    template_vertex_positions: Template vertices, (V, 3).
    identity_basis: Kept identity displacement basis, (I, V, 3).
    expression_basis: Kept expression displacement basis, (E, V, 3).
    template_joint_positions: Template joints, (J, 3).
    joint_identity_basis: Kept joint identity basis, (I, J, 3).
    skinning_weights: Per joint, per vertex skinning weights, (J, V).
    triangles: Triangle vertex indices, (T, 3).
    quad_edges: Unique quad-topology edges, (E, 2).
    component_ids: Per-vertex mesh component index, (V,).
    joint_parent_indices: Parent joint of each joint; root is -1.
  """

  manifest: Mapping[str, Any]
  template_vertex_positions: FloatArray
  identity_basis: FloatArray
  expression_basis: FloatArray
  template_joint_positions: FloatArray
  joint_identity_basis: FloatArray
  skinning_weights: FloatArray
  triangles: IntArray
  quad_edges: IntArray
  component_ids: IntArray
  joint_parent_indices: tuple[int, ...]

  @classmethod
  def load(cls, asset_dir: str) -> 'ReferenceModel':
    """Loads the exported assets from a directory.

    Args:
      asset_dir: Directory holding `gnm_head.bin` and `gnm_head.json`.

    Returns:
      The loaded model.
    """
    with open(os.path.join(asset_dir, 'gnm_head.json'), encoding='utf-8') as f:
      manifest = json.load(f)
    with open(os.path.join(asset_dir, 'gnm_head.bin'), 'rb') as f:
      blob = f.read()

    arrays = _read_views(blob, manifest['arrays'])

    # float16 is a storage format only; every accumulation happens in float32
    # or wider, exactly as the shader will do it.
    def as_float32(name: str) -> FloatArray:
      return arrays[name].astype(np.float32)

    return cls(
        manifest=manifest,
        template_vertex_positions=as_float32('template_vertex_positions'),
        identity_basis=as_float32('identity_basis'),
        expression_basis=as_float32('expression_basis'),
        template_joint_positions=as_float32('template_joint_positions'),
        joint_identity_basis=as_float32('joint_identity_basis'),
        skinning_weights=as_float32('skinning_weights'),
        triangles=arrays['triangles'].astype(np.int32),
        quad_edges=arrays['quad_edges'].astype(np.int32),
        component_ids=arrays['component_ids'].astype(np.int32),
        joint_parent_indices=tuple(manifest['model']['jointParentIndices']),
    )

  @property
  def num_vertices(self) -> int:
    """Number of mesh vertices."""
    return self.template_vertex_positions.shape[0]

  @property
  def num_joints(self) -> int:
    """Number of skeleton joints."""
    return self.template_joint_positions.shape[0]

  @property
  def identity_dim(self) -> int:
    """Number of retained identity components."""
    return self.identity_basis.shape[0]

  @property
  def expression_dim(self) -> int:
    """Number of retained expression components."""
    return self.expression_basis.shape[0]

  def bind_pose_vertices(
      self,
      identity: FloatArray | None = None,
      expression: FloatArray | None = None,
  ) -> FloatArray:
    """Applies the identity and expression bases to the template.

    Args:
      identity: Identity coefficients, (I,), or None for the template.
      expression: Expression coefficients, (E,), or None for neutral.

    Returns:
      Bind-pose vertex positions, (V, 3).
    """
    vertices = self.template_vertex_positions.copy()
    if identity is not None:
      vertices += np.einsum('i,ijk->jk', identity, self.identity_basis)
    if expression is not None:
      vertices += np.einsum('i,ijk->jk', expression, self.expression_basis)
    return vertices

  def bind_pose_joints(self, identity: FloatArray | None = None) -> FloatArray:
    """Applies the identity basis to the template joints.

    Args:
      identity: Identity coefficients, (I,), or None for the template.

    Returns:
      Bind-pose joint positions, (J, 3).
    """
    joints = self.template_joint_positions.copy()
    if identity is not None:
      joints += np.einsum('i,ijk->jk', identity, self.joint_identity_basis)
    return joints

  def joint_transforms_world(
      self,
      joints: FloatArray,
      rotations: FloatArray,
      translation: FloatArray,
  ) -> FloatArray:
    """Runs forward kinematics over the joint hierarchy.

    Args:
      joints: Bind-pose joint positions, (J, 3).
      rotations: Per-joint axis-angle rotations, (J, 3).
      translation: Root translation, (3,).

    Returns:
      World-space joint transforms, (J, 4, 4).
    """
    local_rotations = axis_angle_to_rotation_matrix(rotations)

    # Each joint's local translation is its offset from its parent; the root
    # carries the whole model translation instead.
    parents = np.asarray(self.joint_parent_indices[1:], dtype=np.int32)
    local_translations = np.concatenate(
        [(joints[0] + translation)[None], joints[1:] - joints[parents]],
        axis=0,
    )

    local = np.zeros((self.num_joints, 4, 4), dtype=np.float32)
    local[:, :3, :3] = local_rotations
    local[:, :3, 3] = local_translations
    local[:, 3, 3] = 1.0

    world = [local[0]]
    for joint in range(1, self.num_joints):
      world.append(world[self.joint_parent_indices[joint]] @ local[joint])
    return np.stack(world)

  def linear_blend_skinning(
      self,
      vertices: FloatArray,
      joints: FloatArray,
      rotations: FloatArray,
      translation: FloatArray,
  ) -> FloatArray:
    """Poses bind-pose vertices with linear blend skinning.

    Args:
      vertices: Bind-pose vertex positions, (V, 3).
      joints: Bind-pose joint positions, (J, 3).
      rotations: Per-joint axis-angle rotations, (J, 3).
      translation: Root translation, (3,).

    Returns:
      Posed vertex positions, (V, 3).
    """
    world = self.joint_transforms_world(joints, rotations, translation)

    # Subtracting R_world @ joint from the translation column turns each world
    # transform into one that acts on bind-pose coordinates directly, which is
    # the usual inverse-bind-matrix step written out for a translation-only
    # bind pose.
    transforms = world.copy()
    transforms[:, :3, 3] -= np.einsum('jik,jk->ji', world[:, :3, :3], joints)

    homogeneous = np.concatenate(
        [vertices, np.ones((self.num_vertices, 1), dtype=np.float32)], axis=1
    )
    return np.einsum(
        'jv,jmn,vn->vm', self.skinning_weights, transforms, homogeneous
    )[:, :3]

  def __call__(
      self,
      identity: FloatArray | None = None,
      expression: FloatArray | None = None,
      rotations: FloatArray | None = None,
      translation: FloatArray | None = None,
  ) -> FloatArray:
    """Evaluates the full forward pass.

    Args:
      identity: Identity coefficients, (I,), or None.
      expression: Expression coefficients, (E,), or None.
      rotations: Per-joint axis-angle rotations, (J, 3), or None.
      translation: Root translation, (3,), or None.

    Returns:
      Posed vertex positions, (V, 3).
    """
    if rotations is None:
      rotations = np.zeros((self.num_joints, 3), dtype=np.float32)
    if translation is None:
      translation = np.zeros(3, dtype=np.float32)

    vertices = self.bind_pose_vertices(identity, expression)
    joints = self.bind_pose_joints(identity)
    return self.linear_blend_skinning(
        vertices, joints, rotations, translation
    )

  def expand_identity(self, coefficients: FloatArray) -> FloatArray:
    """Scatters kept identity coefficients back into a full-model vector.

    Args:
      coefficients: Coefficients over the kept components, (I,).

    Returns:
      A full-length identity vector with zeros elsewhere.

    Raises:
      ValueError: If the coefficient count does not match the kept components.
    """
    return self._expand(coefficients, 'identity')

  def expand_expression(self, coefficients: FloatArray) -> FloatArray:
    """Scatters kept expression coefficients back into a full-model vector.

    Args:
      coefficients: Coefficients over the kept components, (E,).

    Returns:
      A full-length expression vector with zeros elsewhere.

    Raises:
      ValueError: If the coefficient count does not match the kept components.
    """
    return self._expand(coefficients, 'expression')

  def _expand(self, coefficients: FloatArray, kind: str) -> FloatArray:
    spec = self.manifest[kind]
    count = spec['count']
    if coefficients.shape != (count,):
      raise ValueError(
          f'Expected {count} {kind} coefficients, '
          f'got shape {coefficients.shape}.'
      )
    full = np.zeros(spec['sourceCount'], dtype=np.float32)
    full[np.asarray(spec['indices'], dtype=np.int32)] = coefficients
    return full
