/**
 * Identity fit: GNM identity coefficients from one frame of tracked landmarks.
 *
 * Like `gnm.ts` this imports nothing, so `node --test` can exercise it against
 * synthetic faces with no browser and no camera involved.
 *
 * The problem is a chicken-and-egg one. Fitting shape needs the head's pose in
 * model space, and estimating that pose needs the shape. So the two alternate:
 * a closed-form similarity fit for the pose, a ridge-regularized linear solve
 * for the shape, repeat. Both halves are exact given the other, and four
 * rounds is well past where either stops moving.
 *
 * Three things make the difference between this working and not, all of them
 * measured rather than assumed:
 *
 * - **The fit is differential.** Landmarks are compared against where the
 *   landmarker puts them on the *neutral* head, not against the GNM vertices
 *   they map to. Those differ by 11 mm rms in z (67 mm at worst): the
 *   landmarker carries its own idea of face shape. Fitting absolute positions
 *   reads that constant bias as a very deep face and bakes it into identity;
 *   fitting displacement cancels it exactly, because it is the same landmarker
 *   on both sides. This is the single change that took the fit from worse than
 *   doing nothing to better than it.
 * - **Only the leading components are solved for.** The normal matrix'
 *   eigenvalues span five orders of magnitude on this point set, so no single
 *   ridge both frees the head of the basis and restrains its tail. The basis is
 *   ordered by variance explained, so truncating is the honest cut.
 * - **Depth is damped, not trusted.** What survives the differential step is
 *   random z error, and it is still the noisiest axis by a wide margin.
 *
 * What this deliberately does not attempt: expression. A single frame cannot
 * separate "this face has a wide mouth" from "this face is smiling", so the
 * solve runs only on the landmarks whose vertices barely move with expression
 * -- the `rigid` mask, derived at export time from expression-basis
 * displacement energy rather than hand-picked. That is 166 of the 473
 * correspondences, which is still 483 equations for the components solved.
 */

import type { GnmModel } from './gnm.ts';

/** The shape of a MediaPipe landmark, without importing the SDK for it. */
export interface Landmark {
  x: number;
  y: number;
  z: number;
}

export interface SimilarityTransform {
  scale: number;
  /** Row-major 3x3. */
  rotation: Float64Array;
  translation: Float64Array;
}

export interface IdentityFitOptions {
  /** Pose/shape alternations. */
  iterations?: number;
  /**
   * Ridge weight, relative to the mean diagonal of the normal matrix.
   *
   * Scaling it to the problem rather than fixing it absolutely is what lets
   * the same number hold when the point count or the model's units change.
   */
  ridge?: number;
  /**
   * Weight on depth residuals, relative to the image-plane ones.
   *
   * MediaPipe's z is not metric -- it is a relative depth in units of roughly
   * the face's own width, and it is the noisiest of the three axes. It still
   * carries real signal about brow and nose projection, so it is damped rather
   * than dropped.
   */
  depthWeight?: number;
  /** Coefficients are clamped to +/- this, matching the sliders' range. */
  limit?: number;
  /**
   * Restrict the solve to the skull-fixed landmarks.
   *
   * Off, all 473 correspondences are used, which conditions the problem far
   * better but lets any expression on the face at capture time be absorbed
   * into identity. See the module comment for the measurements.
   */
  rigidOnly?: boolean;
  /**
   * How many leading identity components to solve for; the rest stay zero.
   *
   * The basis is ordered by variance explained, and sparse landmarks constrain
   * only its head. Truncating is what a single ridge cannot do: the normal
   * matrix' eigenvalues span five orders of magnitude, so any ridge that lets
   * the leading components move freely also lets the tail chase noise.
   */
  components?: number;
}

export interface IdentityFit {
  /** Fitted coefficients, length `model.identityDim`. */
  identity: Float32Array;
  /** The pose that takes observed landmarks into model space. */
  transform: SimilarityTransform;
  /** Landmarks used. */
  points: number;
  /** Identity components solved for; the rest of the vector is zero. */
  components: number;
  /** Residual with the template face, in mm. */
  rmsBefore: number;
  /** Residual after fitting, in mm. */
  rmsAfter: number;
  /** Largest absolute coefficient, as a plausibility check. */
  peak: number;
}

/**
 * Tuned on synthetic faces with 2 px of landmark jitter and depth error at 2%
 * of face width, scored as whole-mesh rms against the face that generated the
 * frame. The template face scores 5.7 mm on that benchmark; these settings
 * score about 1.8 mm.
 */
const DEFAULTS = {
  iterations: 4,
  // 24 leading components. Fewer cannot express the face; more chase noise,
  // and 64 doubles the error at the same ridge.
  components: 24,
  ridge: 0.06,
  // Low, but not zero: dropping depth entirely costs accuracy when z is good,
  // and trusting it costs far more when z is bad.
  depthWeight: 0.1,
  limit: 3,
  rigidOnly: true,
} as const;

/**
 * Converts MediaPipe normalized landmarks into GNM's axis convention.
 *
 * MediaPipe normalizes x by image width and y by image height, so raw values
 * are anisotropic unless the frame is square; its y grows downward and its z
 * away from the camera, both opposite to GNM. The result is defined only up to
 * a global scale -- which the similarity fit solves for -- so all that matters
 * is that the three axes end up sharing one scale.
 *
 * @param landmarks Source landmarks.
 * @param indices Which of them to take, in order.
 * @param aspect Frame width divided by height.
 * @param out Destination, length `indices.length * 3`.
 */
export function landmarksToModelAxes(
  landmarks: readonly Landmark[],
  indices: ArrayLike<number>,
  aspect: number,
  out: Float64Array,
): Float64Array {
  for (let i = 0; i < indices.length; i++) {
    const landmark = landmarks[indices[i]];
    out[i * 3] = landmark.x * aspect;
    out[i * 3 + 1] = -landmark.y;
    // z shares x's normalization, so it takes the same aspect scaling.
    out[i * 3 + 2] = -landmark.z * aspect;
  }
  return out;
}

/**
 * Eigen-decomposition of a small symmetric matrix, by cyclic Jacobi rotations.
 *
 * Only ever called on the 4x4 of `fitSimilarity`, where the iteration
 * converges in a handful of sweeps.
 *
 * @param matrix Row-major symmetric matrix, overwritten.
 * @param n Side length.
 * @returns Eigenvalues, and row-major eigenvectors in columns.
 */
function jacobiEigen(
  matrix: Float64Array,
  n: number,
): { values: Float64Array; vectors: Float64Array } {
  const vectors = new Float64Array(n * n);
  for (let i = 0; i < n; i++) vectors[i * n + i] = 1;

  for (let sweep = 0; sweep < 32; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) off += matrix[p * n + q] * matrix[p * n + q];
    }
    if (off < 1e-24) break;

    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = matrix[p * n + q];
        if (Math.abs(apq) < 1e-18) continue;

        const theta = (matrix[q * n + q] - matrix[p * n + p]) / (2 * apq);
        const t =
          Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;

        for (let k = 0; k < n; k++) {
          const akp = matrix[k * n + p];
          const akq = matrix[k * n + q];
          matrix[k * n + p] = c * akp - s * akq;
          matrix[k * n + q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = matrix[p * n + k];
          const aqk = matrix[q * n + k];
          matrix[p * n + k] = c * apk - s * aqk;
          matrix[q * n + k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = vectors[k * n + p];
          const vkq = vectors[k * n + q];
          vectors[k * n + p] = c * vkp - s * vkq;
          vectors[k * n + q] = s * vkp + c * vkq;
        }
      }
    }
  }

  const values = new Float64Array(n);
  for (let i = 0; i < n; i++) values[i] = matrix[i * n + i];
  return { values, vectors };
}

/**
 * Fits the similarity transform taking source onto target.
 *
 * Minimizes `|| scale * rotation @ source + translation - target ||^2` in
 * closed form, by Horn's quaternion method: the optimal rotation is the
 * eigenvector of largest eigenvalue of a 4x4 built from the cross-covariance.
 * Going through a quaternion rather than an SVD means no reflection case to
 * guard -- a unit quaternion is a rotation by construction.
 *
 * @param source Points, interleaved xyz.
 * @param target Points, interleaved xyz, same length.
 * @throws If the inputs disagree in length or hold fewer than three points.
 */
export function fitSimilarity(
  source: Float64Array,
  target: Float64Array,
): SimilarityTransform {
  if (source.length !== target.length) {
    throw new Error(
      `source and target must be the same length, got ${source.length} and ${target.length}.`,
    );
  }
  const count = source.length / 3;
  if (count < 3) throw new Error(`Need at least 3 points to fit, got ${count}.`);

  let sx = 0, sy = 0, sz = 0, tx = 0, ty = 0, tz = 0;
  for (let i = 0; i < count; i++) {
    sx += source[i * 3]; sy += source[i * 3 + 1]; sz += source[i * 3 + 2];
    tx += target[i * 3]; ty += target[i * 3 + 1]; tz += target[i * 3 + 2];
  }
  sx /= count; sy /= count; sz /= count;
  tx /= count; ty /= count; tz /= count;

  // Cross-covariance of the centred clouds, m[a * 3 + b] = sum(source_a * target_b).
  const m = new Float64Array(9);
  let sourceVariance = 0;
  for (let i = 0; i < count; i++) {
    const ax = source[i * 3] - sx;
    const ay = source[i * 3 + 1] - sy;
    const az = source[i * 3 + 2] - sz;
    const bx = target[i * 3] - tx;
    const by = target[i * 3 + 1] - ty;
    const bz = target[i * 3 + 2] - tz;

    m[0] += ax * bx; m[1] += ax * by; m[2] += ax * bz;
    m[3] += ay * bx; m[4] += ay * by; m[5] += ay * bz;
    m[6] += az * bx; m[7] += az * by; m[8] += az * bz;
    sourceVariance += ax * ax + ay * ay + az * az;
  }

  const [xx, xy, xz, yx, yy, yz, zx, zy, zz] = m;
  const n = Float64Array.from([
    xx + yy + zz, yz - zy, zx - xz, xy - yx,
    yz - zy, xx - yy - zz, xy + yx, zx + xz,
    zx - xz, xy + yx, -xx + yy - zz, yz + zy,
    xy - yx, zx + xz, yz + zy, -xx - yy + zz,
  ]);

  const { values, vectors } = jacobiEigen(n, 4);
  let best = 0;
  for (let i = 1; i < 4; i++) if (values[i] > values[best]) best = i;
  const qw = vectors[best];
  const qx = vectors[4 + best];
  const qy = vectors[8 + best];
  const qz = vectors[12 + best];

  const rotation = Float64Array.from([
    1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy),
    2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx),
    2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy),
  ]);

  // With the rotation known, scale is the ratio of aligned covariance to
  // source spread.
  let aligned = 0;
  for (let i = 0; i < count; i++) {
    const ax = source[i * 3] - sx;
    const ay = source[i * 3 + 1] - sy;
    const az = source[i * 3 + 2] - sz;
    aligned +=
      (target[i * 3] - tx) * (rotation[0] * ax + rotation[1] * ay + rotation[2] * az) +
      (target[i * 3 + 1] - ty) * (rotation[3] * ax + rotation[4] * ay + rotation[5] * az) +
      (target[i * 3 + 2] - tz) * (rotation[6] * ax + rotation[7] * ay + rotation[8] * az);
  }
  const scale = aligned / Math.max(sourceVariance, 1e-12);

  const translation = Float64Array.from([
    tx - scale * (rotation[0] * sx + rotation[1] * sy + rotation[2] * sz),
    ty - scale * (rotation[3] * sx + rotation[4] * sy + rotation[5] * sz),
    tz - scale * (rotation[6] * sx + rotation[7] * sy + rotation[8] * sz),
  ]);

  return { scale, rotation, translation };
}

/** Applies a similarity transform to interleaved xyz points. */
export function applySimilarity(
  points: Float64Array,
  transform: SimilarityTransform,
  out: Float64Array,
): Float64Array {
  const { scale, rotation: r, translation: t } = transform;
  for (let i = 0; i < points.length; i += 3) {
    const x = points[i], y = points[i + 1], z = points[i + 2];
    out[i] = scale * (r[0] * x + r[1] * y + r[2] * z) + t[0];
    out[i + 1] = scale * (r[3] * x + r[4] * y + r[5] * z) + t[1];
    out[i + 2] = scale * (r[6] * x + r[7] * y + r[8] * z) + t[2];
  }
  return out;
}

/** Solves `matrix @ x = rhs` for a symmetric positive-definite matrix. */
export function choleskySolve(matrix: Float64Array, rhs: Float64Array, n: number): Float64Array {
  const l = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = matrix[i * n + j];
      for (let k = 0; k < j; k++) sum -= l[i * n + k] * l[j * n + k];
      if (i === j) {
        if (sum <= 0) throw new Error('Normal matrix is not positive definite.');
        l[i * n + j] = Math.sqrt(sum);
      } else {
        l[i * n + j] = sum / l[j * n + j];
      }
    }
  }

  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let sum = rhs[i];
    for (let k = 0; k < i; k++) sum -= l[i * n + k] * y[k];
    y[i] = sum / l[i * n + i];
  }

  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let sum = y[i];
    for (let k = i + 1; k < n; k++) sum -= l[k * n + i] * x[k];
    x[i] = sum / l[i * n + i];
  }
  return x;
}

/**
 * Fits identity coefficients to one frame of tracked landmarks.
 *
 * @param model The loaded model; must carry a correspondence.
 * @param landmarks One frame's MediaPipe landmarks, normalized.
 * @param aspect Frame width divided by height.
 * @param options Solver knobs; the defaults are what the UI uses.
 * @throws If the model has no correspondence, or the frame is too short.
 */
export function fitIdentity(
  model: GnmModel,
  landmarks: readonly Landmark[],
  aspect: number,
  options: IdentityFitOptions = {},
): IdentityFit {
  const correspondence = model.correspondence;
  if (!correspondence) {
    throw new Error('The model was exported without a correspondence.');
  }
  const { iterations, ridge, depthWeight, limit, rigidOnly, components } = {
    ...DEFAULTS, ...options,
  };

  // Only the skull-fixed landmarks, so an expression on the face at capture
  // time cannot be absorbed into identity.
  const active: number[] = [];
  for (let i = 0; i < correspondence.rigid.length; i++) {
    if (!rigidOnly || correspondence.rigid[i]) active.push(i);
  }
  const count = active.length;
  const needed = Math.max(...correspondence.landmarks) + 1;
  if (landmarks.length < needed) {
    throw new Error(
      `The correspondence indexes landmark ${needed - 1}, but the frame has ${landmarks.length}.`,
    );
  }

  const dim = Math.max(1, Math.min(components, model.identityDim));
  const rows = count * 3;
  const stride = model.vertexCount * 3;

  // Where the landmarker puts these points on a neutral face, and the identity
  // basis restricted to the vertices they map to. Measuring displacement from
  // the reference cloud rather than from the template vertices is what makes
  // this work at all: the landmarker's own face shape sits ~11 mm rms away
  // from GNM's in z, and against the template that bias is indistinguishable
  // from a very deep face, so it lands in the coefficients.
  //
  // Both are constant across iterations, so the normal matrix built from them
  // is too -- only the right-hand side moves.
  const basePoints = new Float64Array(rows);
  const design = new Float64Array(rows * dim);
  for (let i = 0; i < count; i++) {
    const vertex = correspondence.vertices[active[i]];
    for (let axis = 0; axis < 3; axis++) {
      const row = i * 3 + axis;
      basePoints[row] = correspondence.reference[active[i] * 3 + axis];
      for (let k = 0; k < dim; k++) {
        design[row * dim + k] = model.identityBasis[k * stride + vertex * 3 + axis];
      }
    }
  }

  const weights = new Float64Array(rows);
  for (let i = 0; i < count; i++) {
    weights[i * 3] = 1;
    weights[i * 3 + 1] = 1;
    weights[i * 3 + 2] = depthWeight;
  }

  const normal = new Float64Array(dim * dim);
  for (let row = 0; row < rows; row++) {
    const w = weights[row];
    if (w === 0) continue;
    const base = row * dim;
    for (let a = 0; a < dim; a++) {
      const wa = w * design[base + a];
      if (wa === 0) continue;
      for (let b = a; b < dim; b++) {
        normal[a * dim + b] += wa * design[base + b];
      }
    }
  }
  for (let a = 0; a < dim; a++) {
    for (let b = 0; b < a; b++) normal[a * dim + b] = normal[b * dim + a];
  }

  let trace = 0;
  for (let a = 0; a < dim; a++) trace += normal[a * dim + a];
  const lambda = (ridge * trace) / dim;
  for (let a = 0; a < dim; a++) normal[a * dim + a] += lambda;

  const observed = new Float64Array(rows);
  const indices = new Uint16Array(count);
  for (let i = 0; i < count; i++) indices[i] = correspondence.landmarks[active[i]];
  landmarksToModelAxes(landmarks, indices, aspect, observed);

  const identity = new Float64Array(dim);
  const current = new Float64Array(rows);
  const aligned = new Float64Array(rows);
  const rhs = new Float64Array(dim);

  const residualRms = (a: Float64Array, b: Float64Array): number => {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      const d = (a[i] - b[i]) * 1000;
      sum += d * d;
    }
    // Per point, not per scalar: three squared axis errors make one point.
    return Math.sqrt((sum * 3) / a.length);
  };

  let transform: SimilarityTransform | null = null;
  let rmsBefore = 0;
  let rmsAfter = 0;

  for (let iteration = 0; iteration < iterations; iteration++) {
    // Current shape under the running coefficients.
    current.set(basePoints);
    for (let k = 0; k < dim; k++) {
      const c = identity[k];
      if (c === 0) continue;
      for (let row = 0; row < rows; row++) current[row] += c * design[row * dim + k];
    }

    // Pose: the transform taking the observed cloud onto that shape.
    transform = fitSimilarity(observed, current);
    applySimilarity(observed, transform, aligned);
    if (iteration === 0) rmsBefore = residualRms(aligned, basePoints);

    // Shape: the ridge-regularized least squares step onto the aligned cloud.
    rhs.fill(0);
    for (let row = 0; row < rows; row++) {
      const w = weights[row];
      if (w === 0) continue;
      const target = w * (aligned[row] - basePoints[row]);
      const base = row * dim;
      for (let a = 0; a < dim; a++) rhs[a] += design[base + a] * target;
    }

    const solved = choleskySolve(normal, rhs, dim);
    for (let k = 0; k < dim; k++) {
      identity[k] = Math.max(-limit, Math.min(limit, solved[k]));
    }
  }

  // Final residual, measured against the shape the returned coefficients give.
  current.set(basePoints);
  for (let k = 0; k < dim; k++) {
    const c = identity[k];
    if (c === 0) continue;
    for (let row = 0; row < rows; row++) current[row] += c * design[row * dim + k];
  }
  transform = fitSimilarity(observed, current);
  applySimilarity(observed, transform, aligned);
  rmsAfter = residualRms(aligned, current);

  // The untouched tail stays zero, so the result is always the model's full
  // identity vector regardless of how many components were solved for.
  let peak = 0;
  const result = new Float32Array(model.identityDim);
  for (let k = 0; k < dim; k++) {
    result[k] = identity[k];
    peak = Math.max(peak, Math.abs(identity[k]));
  }

  return {
    identity: result,
    transform: transform!,
    points: count,
    components: dim,
    rmsBefore,
    rmsAfter,
    peak,
  };
}
