/**
 * Per-frame retarget: tracked landmarks to GNM expression and head pose.
 *
 * This is the step that finally connects the two halves of the page. M4b fixed
 * who the face is; this one drives what it is doing, every frame.
 *
 * It is the same machinery as the identity fit, pointed at a different basis.
 * Align the tracked cloud onto the subject's own neutral face, and whatever is
 * left over is expression — so the solve is a ridge-regularized projection of
 * that displacement onto the expression basis, and the rotation the alignment
 * removed is the head pose. Because the normal matrix does not depend on the
 * frame, it is built and factored against the subject once, and each frame
 * costs one right-hand side and one back-substitution.
 *
 * Prototype-grade, deliberately:
 *
 * - **Blink is weak.** Eyelid geometry is what defeated the Python prototype's
 *   blink, and nothing here fixes that. The 52 ARKit blendshapes MediaPipe
 *   already reports are the intended fix, and `FaceFrame` carries them; wiring
 *   them to the eye-region components needs a mapping this module does not yet
 *   have.
 * - **Identity must be fit first**, or the subject's own face shape is read as
 *   a permanent expression.
 * - **Only the leading components of a region are solved for**, per
 *   `regionBudget`. Sparse noisy landmarks cannot constrain the rest, and
 *   letting them try inverted the mouth outright. See that option.
 */

import {
  applySimilarity,
  choleskySolve,
  fitSimilarity,
  landmarksToModelAxes,
  type Landmark,
  type SimilarityTransform,
} from './fit.ts';
import type { GnmModel } from './gnm.ts';

export interface RetargetOptions {
  /** Ridge weight, relative to the mean diagonal of the normal matrix. */
  ridge?: number;
  /** Weight on depth residuals; MediaPipe's z is the noisiest axis. */
  depthWeight?: number;
  /** Expression coefficients are clamped to +/- this. */
  limit?: number;
  /**
   * Exponential smoothing in [0, 1). 0 is off; higher is smoother and laggier.
   *
   * A per-frame solve on noisy landmarks visibly shivers, and on a demo whose
   * whole point is how it looks, that reads as broken rather than as honest.
   */
  smoothing?: number;
  /** Multiplier on the solved expression, for demo exaggeration. */
  gain?: number;
  /**
   * Per region, a multiplier on the solved coefficients. Regions not named
   * here are left at 1, and `gain` multiplies on top of all of them.
   *
   * This corrects a measured, near-constant shortfall rather than exaggerating.
   * Handed the model's own vertices the mouth solve returns the true aperture
   * at every amplitude; handed MediaPipe's landmarks it returns 72-77% of it,
   * over a 3x range of opening. Ridge is not the cause -- sweeping it from
   * 0.001 to 0.05 does not move the number by 0.1 mm -- so the deficit is in
   * the landmarks, and it is a scale factor. The per-case optimal gain is 1.29
   * to 1.39, mean 1.35, sd 0.04; at a fixed 1.35 the worst aperture error over
   * the range is 0.22 mm, and nothing reaches the clamp.
   *
   * Only the mouth is corrected. Do not read this as licence to tune the other
   * regions by eye: it is set from a measurement, and the eye regions have no
   * equivalent one yet.
   */
  regionGain?: Readonly<Record<string, number>>;
  /**
   * Per region, how many of its leading components to solve for. Regions not
   * named here are solved in full; components outside the budget stay zero.
   *
   * This is the mouth's most important knob, and it is a *conditioning* limit
   * rather than an expressiveness one. The trailing components of a region are
   * fine detail, and sparse noisy landmarks cannot constrain them -- so least
   * squares spends them fitting landmark noise, in combinations large enough
   * to hit the clamp and wrong enough to invert the gross shape. Measured on
   * rendered mouth openings of 2.5, 5.0 and 7.4 mm, solving all 32 lower-face
   * components returned -2.9, -3.7 and -3.5 mm: the mouth *closed* as the
   * subject opened theirs, which reads as the lips inflating. Four components
   * returned +1.8, +3.8 and +5.4 mm with nothing clamped.
   *
   * The full basis stays in the model, so sliders and the identity fit are
   * unaffected; this restricts only what the per-frame solve may move.
   */
  regionBudget?: Readonly<Record<string, number>>;
}

export interface FaceSolution {
  /** Expression coefficients, length `model.expressionDim`. */
  expression: Float32Array;
  /** Head joint rotation as axis-angle, length 3. */
  headRotation: Float32Array;
  /** How far the tracked cloud sits from the solved face, in mm. */
  rms: number;
  /** The alignment that took the observed landmarks into model space. */
  transform: SimilarityTransform;
}

const DEFAULTS = {
  // Looser than the identity fit's: expression is what we *want* to move, and
  // it is re-solved every frame rather than committed to once.
  ridge: 0.02,
  depthWeight: 0.1,
  limit: 3,
  smoothing: 0.45,
  gain: 1,
  // Only the mouth is capped. The eye regions are the blink problem, which
  // needs the ARKit scores rather than a different basis size, and tongue and
  // pupils are small enough not to matter.
  regionBudget: { lower_face_region: 4 } as Readonly<Record<string, number>>,
  regionGain: { lower_face_region: 1.35 } as Readonly<Record<string, number>>,
} as const;

/** Row-major 3x3 rotation to axis-angle. */
function matrixToAxisAngle(r: Float64Array, out: Float32Array): Float32Array {
  // The trace gives the angle; the antisymmetric part gives the axis. Both
  // degenerate near 0 and pi, and a head pose never reaches pi, so only the
  // small-angle case needs handling.
  const trace = r[0] + r[4] + r[8];
  const cos = Math.min(1, Math.max(-1, (trace - 1) / 2));
  const angle = Math.acos(cos);

  const sin = Math.sin(angle);
  if (Math.abs(sin) < 1e-6) {
    out[0] = 0; out[1] = 0; out[2] = 0;
    return out;
  }
  const scale = angle / (2 * sin);
  out[0] = (r[7] - r[5]) * scale;
  out[1] = (r[2] - r[6]) * scale;
  out[2] = (r[3] - r[1]) * scale;
  return out;
}

export class FaceRetargeter {
  private readonly model: GnmModel;
  private readonly options: Required<RetargetOptions>;

  /** Correspondence indices used for alignment: the skull-fixed ones. */
  private readonly rigid: number[] = [];
  private readonly rigidLandmarks: Uint16Array;
  private readonly allLandmarks: Uint16Array;

  /** Which kept-basis components the solve may move. */
  private readonly active: Uint16Array;
  /** Per active component, its region gain times the global one. */
  private readonly gain: Float64Array;
  private readonly dim: number;
  private readonly rows: number;
  /** Expression basis at the corresponded vertices, rows x dim. */
  private readonly design: Float64Array;
  private readonly weights: Float64Array;
  private readonly normal: Float64Array;

  /** The subject's neutral cloud: reference plus their identity. */
  private readonly base: Float64Array;

  private readonly observed: Float64Array;
  private readonly rigidObserved: Float64Array;
  private readonly rigidBase: Float64Array;
  private readonly aligned: Float64Array;
  private readonly rhs: Float64Array;

  private readonly smoothed: Float32Array;
  private readonly smoothedRotation = new Float32Array(3);
  private started = false;

  /**
   * @param model The loaded model; must carry a correspondence.
   * @param options Solver knobs.
   * @throws If the model was exported without a correspondence.
   */
  constructor(model: GnmModel, options: RetargetOptions = {}) {
    const correspondence = model.correspondence;
    if (!correspondence) {
      throw new Error('The model was exported without a correspondence.');
    }
    this.model = model;
    this.options = { ...DEFAULTS, ...options };

    const count = correspondence.landmarks.length;
    for (let i = 0; i < count; i++) if (correspondence.rigid[i]) this.rigid.push(i);

    this.allLandmarks = correspondence.landmarks;
    this.rigidLandmarks = Uint16Array.from(
      this.rigid.map((i) => correspondence.landmarks[i]),
    );

    // Regions are contiguous slices of the kept basis, so a budget is a
    // prefix of each slice. Anything the manifest does not place in a region
    // is solved for.
    const budgeted = new Uint8Array(model.expressionDim);
    const perComponentGain = new Float64Array(model.expressionDim).fill(1);
    for (const [name, region] of Object.entries(
      model.manifest.expression.regions,
    )) {
      const budget = this.options.regionBudget[name] ?? region.count;
      const keep = Math.min(budget, region.count);
      for (let i = 0; i < keep; i++) budgeted[region.start + i] = 1;
      for (let i = keep; i < region.count; i++) budgeted[region.start + i] = 2;

      const regionGain = this.options.regionGain[name] ?? 1;
      for (let i = 0; i < region.count; i++) {
        perComponentGain[region.start + i] = regionGain;
      }
    }
    const active: number[] = [];
    for (let k = 0; k < model.expressionDim; k++) {
      if (budgeted[k] !== 2) active.push(k);
    }
    this.active = Uint16Array.from(active);
    this.gain = Float64Array.from(
      this.active, (k) => perComponentGain[k] * this.options.gain,
    );

    this.dim = this.active.length;
    this.rows = count * 3;
    const stride = model.vertexCount * 3;

    // Expression is fit on *all* the correspondences, unlike identity: the
    // mouth and brow are exactly the landmarks that move, so excluding them
    // would leave nothing to solve.
    this.design = new Float64Array(this.rows * this.dim);
    for (let i = 0; i < count; i++) {
      const vertex = correspondence.vertices[i];
      for (let axis = 0; axis < 3; axis++) {
        const row = i * 3 + axis;
        for (let k = 0; k < this.dim; k++) {
          this.design[row * this.dim + k] =
            model.expressionBasis[this.active[k] * stride + vertex * 3 + axis];
        }
      }
    }

    this.weights = new Float64Array(this.rows);
    for (let i = 0; i < count; i++) {
      this.weights[i * 3] = 1;
      this.weights[i * 3 + 1] = 1;
      this.weights[i * 3 + 2] = this.options.depthWeight;
    }

    this.normal = new Float64Array(this.dim * this.dim);
    for (let row = 0; row < this.rows; row++) {
      const w = this.weights[row];
      const base = row * this.dim;
      for (let a = 0; a < this.dim; a++) {
        const wa = w * this.design[base + a];
        if (wa === 0) continue;
        for (let b = a; b < this.dim; b++) {
          this.normal[a * this.dim + b] += wa * this.design[base + b];
        }
      }
    }
    for (let a = 0; a < this.dim; a++) {
      for (let b = 0; b < a; b++) this.normal[a * this.dim + b] = this.normal[b * this.dim + a];
    }
    let trace = 0;
    for (let a = 0; a < this.dim; a++) trace += this.normal[a * this.dim + a];
    const lambda = (this.options.ridge * trace) / this.dim;
    for (let a = 0; a < this.dim; a++) this.normal[a * this.dim + a] += lambda;

    this.base = new Float64Array(this.rows);
    this.observed = new Float64Array(this.rows);
    this.rigidObserved = new Float64Array(this.rigid.length * 3);
    this.rigidBase = new Float64Array(this.rigid.length * 3);
    this.aligned = new Float64Array(this.rows);
    this.rhs = new Float64Array(this.dim);
    // Full length, so callers still get one coefficient per kept component;
    // the ones outside the budget simply stay zero.
    this.smoothed = new Float32Array(model.expressionDim);

    this.setIdentity(new Float32Array(model.identityDim));
  }

  /**
   * Rebuilds the neutral cloud this subject's expressions are measured from.
   *
   * Call whenever identity changes. Skipping it means the subject's own face
   * shape shows up as a constant expression -- a permanent smirk on anyone
   * whose mouth differs from the template's.
   */
  setIdentity(identity: ArrayLike<number>): void {
    const correspondence = this.model.correspondence!;
    const stride = this.model.vertexCount * 3;
    const count = correspondence.landmarks.length;

    for (let i = 0; i < count; i++) {
      const vertex = correspondence.vertices[i];
      for (let axis = 0; axis < 3; axis++) {
        let value = correspondence.reference[i * 3 + axis];
        for (let k = 0; k < identity.length; k++) {
          const c = identity[k];
          if (c === 0) continue;
          value += c * this.model.identityBasis[k * stride + vertex * 3 + axis];
        }
        this.base[i * 3 + axis] = value;
      }
    }

    for (let i = 0; i < this.rigid.length; i++) {
      const source = this.rigid[i] * 3;
      this.rigidBase[i * 3] = this.base[source];
      this.rigidBase[i * 3 + 1] = this.base[source + 1];
      this.rigidBase[i * 3 + 2] = this.base[source + 2];
    }
  }

  /** Drops the smoothing history, so the next frame is taken as-is. */
  reset(): void {
    this.started = false;
    this.smoothed.fill(0);
    this.smoothedRotation.fill(0);
  }

  /**
   * Solves one frame.
   *
   * @param landmarks MediaPipe landmarks, normalized.
   * @param aspect Frame width divided by height.
   * @returns The expression and head pose for this frame, smoothed.
   */
  solve(landmarks: readonly Landmark[], aspect: number): FaceSolution {
    landmarksToModelAxes(landmarks, this.allLandmarks, aspect, this.observed);
    landmarksToModelAxes(landmarks, this.rigidLandmarks, aspect, this.rigidObserved);

    // Align on the skull-fixed points only. Aligning on all of them would let
    // an open mouth drag the head's estimated pose around with it.
    const transform = fitSimilarity(this.rigidObserved, this.rigidBase);
    applySimilarity(this.observed, transform, this.aligned);

    this.rhs.fill(0);
    for (let row = 0; row < this.rows; row++) {
      const w = this.weights[row];
      const target = w * (this.aligned[row] - this.base[row]);
      if (target === 0) continue;
      const base = row * this.dim;
      for (let a = 0; a < this.dim; a++) this.rhs[a] += this.design[base + a] * target;
    }

    const solved = choleskySolve(this.normal, this.rhs, this.dim);

    const { limit, smoothing } = this.options;
    const alpha = this.started ? 1 - smoothing : 1;
    for (let k = 0; k < this.dim; k++) {
      const target = this.active[k];
      const value = Math.max(-limit, Math.min(limit, solved[k] * this.gain[k]));
      this.smoothed[target] += (value - this.smoothed[target]) * alpha;
    }

    // The alignment rotates the observed cloud onto the neutral model, so the
    // head's own rotation is its inverse -- which for a rotation is the
    // transpose.
    const r = transform.rotation;
    const inverse = Float64Array.from([r[0], r[3], r[6], r[1], r[4], r[7], r[2], r[5], r[8]]);
    const rotation = matrixToAxisAngle(inverse, new Float32Array(3));
    for (let i = 0; i < 3; i++) {
      this.smoothedRotation[i] += (rotation[i] - this.smoothedRotation[i]) * alpha;
    }
    this.started = true;

    // Residual against the smoothed answer, so the readout reflects what is
    // actually on screen rather than the unsmoothed solve.
    let sum = 0;
    for (let row = 0; row < this.rows; row++) {
      let predicted = this.base[row];
      const base = row * this.dim;
      for (let k = 0; k < this.dim; k++) {
        predicted += this.smoothed[this.active[k]] * this.design[base + k];
      }
      const d = (this.aligned[row] - predicted) * 1000;
      sum += d * d;
    }
    const rms = Math.sqrt((sum * 3) / this.rows);

    return {
      expression: this.smoothed,
      headRotation: this.smoothedRotation,
      rms,
      transform,
    };
  }
}
