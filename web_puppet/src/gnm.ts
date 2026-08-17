/**
 * The GNM head forward pass, in TypeScript.
 *
 * This file deliberately imports nothing. It is the piece that has to agree
 * with `tools/reference_model.py`, which is itself checked against
 * `gnm.shape`, so keeping it free of Three.js and of the DOM is what lets
 * `node --test` verify it against frozen golden vectors with no browser
 * involved.
 *
 * The evaluation here runs on the CPU. That is the right cost for M1, where
 * shape changes only when a slider moves; M2 moves it into a vertex shader
 * for the per-frame case.
 */

export interface ArrayView {
  dtype: 'float32' | 'float16' | 'uint16' | 'uint8';
  shape: number[];
  byteOffset: number;
  byteLength: number;
}

export interface Manifest {
  model: {
    version: string;
    variant: string;
    vertexCount: number;
    jointCount: number;
    jointNames: string[];
    jointParentIndices: number[];
    componentNames: string[];
    maxNormalCorners: number;
  };
  correspondence: { count: number };
  /**
   * Authored shapes the GNM basis cannot reach, each driven by the MediaPipe
   * blendshape of the same position in `blendshapes`. Not model output -- see
   * `cheek_puff_corrective` in the export tool for why they exist.
   */
  correctives?: { names: string[]; blendshapes: string[] };
  identity: { count: number; sourceCount: number; indices: number[]; names: string[] };
  expression: {
    count: number;
    sourceCount: number;
    indices: number[];
    names: string[];
    regions: Record<string, { start: number; count: number }>;
  };
  arrays: Record<string, ArrayView>;
}

/**
 * Every float16 bit pattern, decoded once.
 *
 * The bases arrive as raw half-float bits and are used as plain numbers on the
 * CPU path, so they need decoding. Doing it per value costs a branch and a
 * `Math.pow` on eight million entries; a 128 KB table turns the whole thing
 * into an array index. The table is built via the exponent trick rather than
 * arithmetic: a half's mantissa and exponent can be repositioned into a
 * float32 bit pattern directly.
 */
const HALF_TO_FLOAT = /* @__PURE__ */ (() => {
  const table = new Float32Array(65536);
  const view = new DataView(new ArrayBuffer(4));
  for (let bits = 0; bits < 65536; bits++) {
    const sign = (bits & 0x8000) << 16;
    const exponent = (bits & 0x7c00) >> 10;
    const mantissa = bits & 0x03ff;

    let out: number;
    if (exponent === 0) {
      // Subnormal: no implicit leading 1, so it cannot be expressed by
      // shifting the exponent and has to be scaled explicitly.
      out = 0;
      view.setFloat32(0, mantissa * 5.960464477539063e-8);
      out = view.getFloat32(0) * (bits & 0x8000 ? -1 : 1);
      table[bits] = out;
      continue;
    }
    if (exponent === 0x1f) {
      table[bits] = mantissa ? NaN : (bits & 0x8000 ? -Infinity : Infinity);
      continue;
    }
    // Rebias the exponent from half (15) to single (127) and left-align the
    // mantissa from 10 bits to 23.
    view.setUint32(0, sign | ((exponent - 15 + 127) << 23) | (mantissa << 13));
    table[bits] = view.getFloat32(0);
  }
  return table;
})();

function decodeHalf(bits: Uint16Array): Float32Array {
  const out = new Float32Array(bits.length);
  for (let i = 0; i < bits.length; i++) out[i] = HALF_TO_FLOAT[bits[i]];
  return out;
}

function view(buffer: ArrayBuffer, spec: ArrayView): Float32Array | Uint16Array | Uint8Array {
  switch (spec.dtype) {
    case 'float32':
      return new Float32Array(buffer, spec.byteOffset, spec.byteLength / 4);
    case 'float16':
    case 'uint16':
      return new Uint16Array(buffer, spec.byteOffset, spec.byteLength / 2);
    case 'uint8':
      return new Uint8Array(buffer, spec.byteOffset, spec.byteLength);
  }
}

function floats(buffer: ArrayBuffer, spec: ArrayView): Float32Array {
  const raw = view(buffer, spec);
  if (spec.dtype === 'float16') return decodeHalf(raw as Uint16Array);
  if (spec.dtype === 'float32') return raw as Float32Array;
  throw new Error(`Expected a float array, got ${spec.dtype}.`);
}

/** Writes the 3x3 Rodrigues rotation of an axis-angle vector into `out`. */
function axisAngleToMatrix(x: number, y: number, z: number, out: Float32Array, at: number): void {
  const angle = Math.hypot(x, y, z);
  if (angle < 1e-12) {
    out[at] = 1; out[at + 1] = 0; out[at + 2] = 0;
    out[at + 3] = 0; out[at + 4] = 1; out[at + 5] = 0;
    out[at + 6] = 0; out[at + 7] = 0; out[at + 8] = 1;
    return;
  }
  const ax = x / angle, ay = y / angle, az = z / angle;
  const s = Math.sin(angle);
  const c = Math.cos(angle);
  const t = 1 - c;

  out[at] = t * ax * ax + c;
  out[at + 1] = t * ax * ay - s * az;
  out[at + 2] = t * ax * az + s * ay;
  out[at + 3] = t * ax * ay + s * az;
  out[at + 4] = t * ay * ay + c;
  out[at + 5] = t * ay * az - s * ax;
  out[at + 6] = t * ax * az - s * ay;
  out[at + 7] = t * ay * az + s * ax;
  out[at + 8] = t * az * az + c;
}

export class GnmModel {
  readonly manifest: Manifest;
  readonly vertexCount: number;
  readonly jointCount: number;
  readonly identityDim: number;
  readonly expressionDim: number;

  readonly template: Float32Array;
  readonly identityBasis: Float32Array;
  readonly expressionBasis: Float32Array;
  /** Corrective displacement basis, `correctiveDim` x vertexCount x 3. */
  readonly correctiveBasis: Float32Array;
  readonly correctiveDim: number;
  /** MediaPipe blendshape driving each corrective, in the same order. */
  readonly correctiveBlendshapes: string[];
  readonly templateJoints: Float32Array;
  readonly jointIdentityBasis: Float32Array;
  readonly skinningWeights: Float32Array;
  readonly triangles: Uint16Array;
  readonly quadEdges: Uint16Array;
  readonly componentIds: Uint8Array;
  readonly normalAdjacency: Uint16Array;
  readonly normalAdjacencyCount: Uint8Array;
  readonly maxNormalCorners: number;
  readonly jointParents: number[];
  /** Kept so the GPU path can upload the float16 bases without re-encoding. */
  readonly buffer: ArrayBuffer;

  /**
   * MediaPipe landmark to GNM vertex mapping, or null if not exported.
   *
   * `landmarks[i]` is a MediaPipe landmark index and `vertices[i]` the GNM
   * vertex it lands on. `colors[i]` identifies the pair in both views: the
   * whole reason to draw these twice is to be able to say "that point is that
   * point", which needs the two drawings to agree on a colour.
   *
   * `reference[i]` is where the landmarker places that landmark on the neutral
   * head, in metric model space -- which is *not* the vertex it maps to. The
   * landmarker carries its own idea of face shape, biased against GNM by about
   * 11 mm rms in z, and the identity fit measures displacement from this cloud
   * so that bias cancels rather than being absorbed as identity.
   */
  readonly correspondence: {
    landmarks: Uint16Array;
    vertices: Uint16Array;
    rigid: Uint8Array;
    reference: Float32Array;
    colors: Uint8Array;
  } | null;

  private readonly bindVertices: Float32Array;
  private readonly bindJoints: Float32Array;
  private readonly transforms: Float32Array;

  constructor(manifest: Manifest, buffer: ArrayBuffer) {
    this.manifest = manifest;
    const a = manifest.arrays;

    this.template = floats(buffer, a.template_vertex_positions);
    this.identityBasis = floats(buffer, a.identity_basis);
    this.expressionBasis = floats(buffer, a.expression_basis);
    // Optional: an export predating the correctives is still loadable, and the
    // shader compiles its loop away when the count is zero.
    this.correctiveBasis = a.corrective_basis
      ? floats(buffer, a.corrective_basis)
      : new Float32Array(0);
    this.correctiveBlendshapes = manifest.correctives?.blendshapes ?? [];
    this.correctiveDim = this.correctiveBlendshapes.length;
    this.templateJoints = floats(buffer, a.template_joint_positions);
    this.jointIdentityBasis = floats(buffer, a.joint_identity_basis);
    this.skinningWeights = floats(buffer, a.skinning_weights);
    this.triangles = view(buffer, a.triangles) as Uint16Array;
    this.quadEdges = view(buffer, a.quad_edges) as Uint16Array;
    this.componentIds = view(buffer, a.component_ids) as Uint8Array;
    this.normalAdjacency = view(buffer, a.normal_adjacency) as Uint16Array;
    this.normalAdjacencyCount = view(
      buffer, a.normal_adjacency_count,
    ) as Uint8Array;
    this.maxNormalCorners = manifest.model.maxNormalCorners;
    this.buffer = buffer;

    this.correspondence = a.correspondence_landmarks
      ? {
          landmarks: view(buffer, a.correspondence_landmarks) as Uint16Array,
          vertices: view(buffer, a.correspondence_vertices) as Uint16Array,
          rigid: view(buffer, a.correspondence_rigid) as Uint8Array,
          reference: floats(buffer, a.correspondence_reference),
          colors: view(buffer, a.correspondence_colors) as Uint8Array,
        }
      : null;

    this.vertexCount = manifest.model.vertexCount;
    this.jointCount = manifest.model.jointCount;
    this.identityDim = manifest.identity.count;
    this.expressionDim = manifest.expression.count;
    this.jointParents = manifest.model.jointParentIndices;

    this.bindVertices = new Float32Array(this.vertexCount * 3);
    this.bindJoints = new Float32Array(this.jointCount * 3);
    // One 3x4 matrix per joint, row-major.
    this.transforms = new Float32Array(this.jointCount * 12);
  }

  /**
   * Adds a basis' contribution to `out`, skipping zero coefficients.
   *
   * Most coefficients are zero most of the time -- a handful of sliders are
   * ever off-centre -- and each non-zero one costs a full pass over 53k
   * floats, so the skip is what keeps slider dragging interactive.
   */
  private accumulate(coefficients: ArrayLike<number>, basis: Float32Array, out: Float32Array): void {
    const stride = out.length;
    for (let i = 0; i < coefficients.length; i++) {
      const c = coefficients[i];
      if (c > -1e-8 && c < 1e-8) continue;
      const base = i * stride;
      for (let k = 0; k < stride; k++) out[k] += c * basis[base + k];
    }
  }

  /**
   * Applies the identity, expression and corrective bases to the template.
   *
   * Correctives are added here, in the bind pose, rather than to the posed
   * result: skinning has to carry them, or a puffed cheek would stay behind
   * when the head turns.
   */
  bindPose(
    identity: ArrayLike<number>,
    expression: ArrayLike<number>,
    correctives?: ArrayLike<number>,
  ): Float32Array {
    this.bindVertices.set(this.template);
    this.accumulate(identity, this.identityBasis, this.bindVertices);
    this.accumulate(expression, this.expressionBasis, this.bindVertices);
    if (correctives) {
      this.accumulate(correctives, this.correctiveBasis, this.bindVertices);
    }
    return this.bindVertices;
  }

  /**
   * Runs forward kinematics and folds the bind pose into each joint transform.
   *
   * Subtracting `R_world @ joint` from the translation column produces a
   * matrix that acts on bind-pose coordinates directly, which is the
   * inverse-bind-matrix step written out for a translation-only bind pose.
   */
  private poseJoints(rotations: ArrayLike<number>, translation: ArrayLike<number>): void {
    const joints = this.bindJoints;
    const rotation = new Float32Array(9);
    const world = new Float32Array(this.jointCount * 12);

    for (let j = 0; j < this.jointCount; j++) {
      axisAngleToMatrix(rotations[j * 3], rotations[j * 3 + 1], rotations[j * 3 + 2], rotation, 0);

      // Local translation: offset from the parent, or the model translation
      // at the root.
      const parent = this.jointParents[j];
      let tx: number, ty: number, tz: number;
      if (parent < 0) {
        tx = joints[0] + translation[0];
        ty = joints[1] + translation[1];
        tz = joints[2] + translation[2];
      } else {
        tx = joints[j * 3] - joints[parent * 3];
        ty = joints[j * 3 + 1] - joints[parent * 3 + 1];
        tz = joints[j * 3 + 2] - joints[parent * 3 + 2];
      }

      const o = j * 12;
      if (parent < 0) {
        for (let r = 0; r < 3; r++) {
          world[o + r * 4] = rotation[r * 3];
          world[o + r * 4 + 1] = rotation[r * 3 + 1];
          world[o + r * 4 + 2] = rotation[r * 3 + 2];
        }
        world[o + 3] = tx; world[o + 7] = ty; world[o + 11] = tz;
      } else {
        const p = parent * 12;
        for (let r = 0; r < 3; r++) {
          for (let c = 0; c < 3; c++) {
            world[o + r * 4 + c] =
              world[p + r * 4] * rotation[c] +
              world[p + r * 4 + 1] * rotation[3 + c] +
              world[p + r * 4 + 2] * rotation[6 + c];
          }
          world[o + r * 4 + 3] =
            world[p + r * 4] * tx +
            world[p + r * 4 + 1] * ty +
            world[p + r * 4 + 2] * tz +
            world[p + r * 4 + 3];
        }
      }
    }

    this.transforms.set(world);
    for (let j = 0; j < this.jointCount; j++) {
      const o = j * 12;
      const jx = joints[j * 3], jy = joints[j * 3 + 1], jz = joints[j * 3 + 2];
      for (let r = 0; r < 3; r++) {
        this.transforms[o + r * 4 + 3] -=
          world[o + r * 4] * jx + world[o + r * 4 + 1] * jy + world[o + r * 4 + 2] * jz;
      }
    }
  }

  /**
   * The per-joint matrices that take bind-pose coordinates to posed ones.
   *
   * Four 3x4 matrices is the entire per-frame cost of the skeleton, so this
   * stays on the CPU even in the GPU path and is uploaded as uniforms.
   *
   * @param identity Identity coefficients, length `identityDim`.
   * @param rotations Per-joint axis-angle rotations, length `jointCount * 3`.
   * @param translation Root translation, length 3.
   * @returns Row-major 3x4 matrices, length `jointCount * 12`.
   */
  jointTransforms(
    identity: ArrayLike<number>,
    rotations: ArrayLike<number>,
    translation: ArrayLike<number>,
  ): Float32Array {
    this.bindJoints.set(this.templateJoints);
    this.accumulate(identity, this.jointIdentityBasis, this.bindJoints);
    this.poseJoints(rotations, translation);
    return this.transforms;
  }

  /**
   * Evaluates the full forward pass.
   *
   * @param identity Identity coefficients, length `identityDim`.
   * @param expression Expression coefficients, length `expressionDim`.
   * @param rotations Per-joint axis-angle rotations, length `jointCount * 3`.
   * @param translation Root translation, length 3.
   * @param out Optional destination, length `vertexCount * 3`.
   * @param correctives Corrective weights, length `correctiveDim`.
   * @returns Posed vertex positions, interleaved xyz.
   */
  evaluate(
    identity: ArrayLike<number>,
    expression: ArrayLike<number>,
    rotations: ArrayLike<number>,
    translation: ArrayLike<number>,
    out?: Float32Array,
    correctives?: ArrayLike<number>,
  ): Float32Array {
    const result = out ?? new Float32Array(this.vertexCount * 3);

    const vertices = this.bindPose(identity, expression, correctives);

    this.bindJoints.set(this.templateJoints);
    this.accumulate(identity, this.jointIdentityBasis, this.bindJoints);
    this.poseJoints(rotations, translation);

    const weights = this.skinningWeights;
    const transforms = this.transforms;
    const vertexCount = this.vertexCount;

    result.fill(0);
    for (let j = 0; j < this.jointCount; j++) {
      const o = j * 12;
      const m0 = transforms[o], m1 = transforms[o + 1], m2 = transforms[o + 2], m3 = transforms[o + 3];
      const m4 = transforms[o + 4], m5 = transforms[o + 5], m6 = transforms[o + 6], m7 = transforms[o + 7];
      const m8 = transforms[o + 8], m9 = transforms[o + 9], m10 = transforms[o + 10], m11 = transforms[o + 11];
      const base = j * vertexCount;

      for (let v = 0; v < vertexCount; v++) {
        const w = weights[base + v];
        if (w === 0) continue;
        const i = v * 3;
        const x = vertices[i], y = vertices[i + 1], z = vertices[i + 2];
        result[i] += w * (m0 * x + m1 * y + m2 * z + m3);
        result[i + 1] += w * (m4 * x + m5 * y + m6 * z + m7);
        result[i + 2] += w * (m8 * x + m9 * y + m10 * z + m11);
      }
    }
    return result;
  }

  /**
   * Computes unit vertex normals by area-weighted face-normal accumulation.
   *
   * The un-normalized cross product is twice the triangle area, so
   * accumulating it directly area-weights each face's contribution for free.
   */
  computeNormals(vertices: Float32Array, out?: Float32Array): Float32Array {
    const normals = out ?? new Float32Array(this.vertexCount * 3);
    normals.fill(0);

    const triangles = this.triangles;
    for (let t = 0; t < triangles.length; t += 3) {
      const ia = triangles[t] * 3, ib = triangles[t + 1] * 3, ic = triangles[t + 2] * 3;

      const abx = vertices[ib] - vertices[ia];
      const aby = vertices[ib + 1] - vertices[ia + 1];
      const abz = vertices[ib + 2] - vertices[ia + 2];
      const acx = vertices[ic] - vertices[ia];
      const acy = vertices[ic + 1] - vertices[ia + 1];
      const acz = vertices[ic + 2] - vertices[ia + 2];

      const nx = aby * acz - abz * acy;
      const ny = abz * acx - abx * acz;
      const nz = abx * acy - aby * acx;

      normals[ia] += nx; normals[ia + 1] += ny; normals[ia + 2] += nz;
      normals[ib] += nx; normals[ib + 1] += ny; normals[ib + 2] += nz;
      normals[ic] += nx; normals[ic + 1] += ny; normals[ic + 2] += nz;
    }

    for (let i = 0; i < normals.length; i += 3) {
      const length = Math.hypot(normals[i], normals[i + 1], normals[i + 2]) || 1;
      normals[i] /= length; normals[i + 1] /= length; normals[i + 2] /= length;
    }
    return normals;
  }

  /**
   * Reorders triangles so each mesh component occupies one contiguous run.
   *
   * Three.js draws a multi-material mesh as ranges of the index buffer, so the
   * eyes can only take a different material than the skin if their triangles
   * are adjacent in it.
   *
   * @returns The reordered index buffer and one `{start, count}` per component.
   */
  groupTrianglesByComponent(): {
    indices: Uint16Array;
    groups: { component: number; start: number; count: number }[];
  } {
    const componentCount = this.manifest.model.componentNames.length;
    const buckets: number[][] = Array.from({ length: componentCount }, () => []);

    for (let t = 0; t < this.triangles.length; t += 3) {
      const a = this.componentIds[this.triangles[t]];
      const b = this.componentIds[this.triangles[t + 1]];
      const c = this.componentIds[this.triangles[t + 2]];
      if (a !== b || b !== c) {
        throw new Error(`Triangle ${t / 3} spans components ${a}/${b}/${c}.`);
      }
      buckets[a].push(this.triangles[t], this.triangles[t + 1], this.triangles[t + 2]);
    }

    const indices = new Uint16Array(this.triangles.length);
    const groups: { component: number; start: number; count: number }[] = [];
    let offset = 0;
    for (let component = 0; component < componentCount; component++) {
      const bucket = buckets[component];
      if (bucket.length === 0) continue;
      indices.set(bucket, offset);
      groups.push({ component, start: offset, count: bucket.length });
      offset += bucket.length;
    }
    return { indices, groups };
  }
}

/** Fetches and constructs the model from a base URL holding the assets. */
export async function loadGnmModel(baseUrl: string): Promise<GnmModel> {
  const [manifest, buffer] = await Promise.all([
    fetch(`${baseUrl}/gnm_head.json`).then((r) => r.json() as Promise<Manifest>),
    fetch(`${baseUrl}/gnm_head.bin`).then((r) => r.arrayBuffer()),
  ]);
  return new GnmModel(manifest, buffer);
}
