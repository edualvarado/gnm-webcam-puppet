/**
 * Checks the per-frame retarget against frames whose answer is known.
 *
 * Same trick as `fit.test.ts`: the model generates the landmarks a face with a
 * given identity, expression and head pose would produce, and the solver has
 * to get back to them.
 *
 * Run from web_puppet/:
 *   npm test
 */

import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import test from 'node:test';

import { applySimilarity, type Landmark, type SimilarityTransform } from './fit.ts';
import { GnmModel, type Manifest } from './gnm.ts';
import { FaceRetargeter } from './retarget.ts';

const here = dirname(fileURLToPath(import.meta.url));
const assetDir = join(here, '..', 'public', 'assets');
const ASPECT = 1280 / 720;

async function loadBuffer(path: string): Promise<ArrayBuffer> {
  const data = await readFile(path);
  return data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) as ArrayBuffer;
}

const manifest = JSON.parse(
  await readFile(join(assetDir, 'gnm_head.json'), 'utf8'),
) as Manifest;
const model = new GnmModel(manifest, await loadBuffer(join(assetDir, 'gnm_head.bin')));
const correspondence = model.correspondence!;
const stride = model.vertexCount * 3;

function rotationY(angle: number): Float64Array {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return Float64Array.from([c, 0, s, 0, 1, 0, -s, 0, c]);
}

/**
 * The landmark frame a face with this identity, expression and pose produces.
 *
 * Built from the reference cloud plus basis displacements rather than from
 * `bindPose`, both to model where the landmarker actually puts its points and
 * to avoid `bindPose`'s shared return buffer.
 */
function syntheticFrame(
  identity: ArrayLike<number>,
  expression: ArrayLike<number>,
  transform: SimilarityTransform,
): Landmark[] {
  const count = correspondence.landmarks.length;
  const points = new Float64Array(count * 3);

  for (let i = 0; i < count; i++) {
    const vertex = correspondence.vertices[i];
    for (let axis = 0; axis < 3; axis++) {
      let value = correspondence.reference[i * 3 + axis];
      for (let k = 0; k < identity.length; k++) {
        if (identity[k]) value += identity[k] * model.identityBasis[k * stride + vertex * 3 + axis];
      }
      for (let k = 0; k < expression.length; k++) {
        if (expression[k]) {
          value += expression[k] * model.expressionBasis[k * stride + vertex * 3 + axis];
        }
      }
      points[i * 3 + axis] = value;
    }
  }

  const placed = applySimilarity(points, transform, new Float64Array(points.length));
  const highest = Math.max(...correspondence.landmarks) + 1;
  const landmarks: Landmark[] = Array.from({ length: highest }, () => ({ x: 0, y: 0, z: 0 }));
  for (let i = 0; i < count; i++) {
    landmarks[correspondence.landmarks[i]] = {
      x: placed[i * 3] / ASPECT,
      y: -placed[i * 3 + 1],
      z: -placed[i * 3 + 2] / ASPECT,
    };
  }
  return landmarks;
}

const STILL: SimilarityTransform = {
  scale: 3.9,
  rotation: rotationY(0),
  translation: Float64Array.from([0.05, -0.02, 0.3]),
};

/** A mouth-region expression, as a stand-in for "the subject is talking". */
function mouthExpression(): Float32Array {
  const expression = new Float32Array(model.expressionDim);
  const region = manifest.expression.regions['lower_face_region'];
  const amounts = [1.4, -0.9, 1.1, 0.6];
  for (let i = 0; i < amounts.length; i++) expression[region.start + i] = amounts[i];
  return expression;
}

test('a resting face solves as no expression and no head rotation', () => {
  const retargeter = new FaceRetargeter(model, { smoothing: 0 });
  const identity = new Float32Array(model.identityDim);
  identity[0] = 0.8;
  identity[3] = -1.1;
  retargeter.setIdentity(identity);

  const solution = retargeter.solve(
    syntheticFrame(identity, new Float32Array(model.expressionDim), STILL),
    ASPECT,
  );

  let peak = 0;
  for (const value of solution.expression) peak = Math.max(peak, Math.abs(value));
  const rotation = Math.hypot(...solution.headRotation);
  console.log(
    `    resting: rms ${solution.rms.toFixed(4)} mm, peak |expression| ${peak.toFixed(4)}, ` +
      `head ${rotation.toFixed(4)} rad`,
  );

  assert.ok(solution.rms < 0.05, `rms ${solution.rms} mm on a resting face`);
  assert.ok(peak < 0.05, `invented expression ${peak} on a resting face`);
  assert.ok(rotation < 0.01, `invented head rotation ${rotation} rad`);
});

test('identity is not read as expression', () => {
  // The failure this guards: forget to call setIdentity and the subject's own
  // face shape becomes a permanent expression.
  const identity = new Float32Array(model.identityDim);
  identity[0] = 1.5;
  identity[1] = -1.2;
  identity[2] = 0.9;
  const frame = syntheticFrame(identity, new Float32Array(model.expressionDim), STILL);

  const informed = new FaceRetargeter(model, { smoothing: 0 });
  informed.setIdentity(identity);
  let informedPeak = 0;
  for (const v of informed.solve(frame, ASPECT).expression) {
    informedPeak = Math.max(informedPeak, Math.abs(v));
  }

  const ignorant = new FaceRetargeter(model, { smoothing: 0 });
  let ignorantPeak = 0;
  for (const v of ignorant.solve(frame, ASPECT).expression) {
    ignorantPeak = Math.max(ignorantPeak, Math.abs(v));
  }

  console.log(
    `    peak |expression| with identity ${informedPeak.toFixed(3)}, without ${ignorantPeak.toFixed(3)}`,
  );
  assert.ok(informedPeak < 0.05, `identity leaked into expression: ${informedPeak}`);
  assert.ok(ignorantPeak > informedPeak * 5, 'the guard case should misbehave, and did not');
});

test('an expression is recovered from the landmarks it moves', () => {
  // These frames are generated by the model, so the observations are perfect and
  // the landmark shortfall `regionGain` compensates for does not exist. Leaving it
  // on would measure the compensation rather than the solve.
  const retargeter = new FaceRetargeter(model, { smoothing: 0, regionGain: {} });
  const identity = new Float32Array(model.identityDim);
  const truth = mouthExpression();

  const solution = retargeter.solve(syntheticFrame(identity, truth, STILL), ASPECT);

  let worst = 0;
  const region = manifest.expression.regions['lower_face_region'];
  for (let i = 0; i < 4; i++) {
    worst = Math.max(worst, Math.abs(solution.expression[region.start + i] - truth[region.start + i]));
  }
  console.log(`    expression: rms ${solution.rms.toFixed(3)} mm, worst coefficient ${worst.toFixed(3)}`);

  // Unlike identity, expression *is* recoverable as coefficients: the basis is
  // region-blocked and the landmarks that move are exactly the ones fit. What
  // remains is ridge shrinkage, which is deliberate -- the coefficients come
  // back slightly short of the truth rather than slightly wrong, and a demo
  // that under-reacts is worth more than one that twitches.
  assert.ok(solution.rms < 0.3, `rms ${solution.rms} mm`);
  assert.ok(worst < 0.2, `worst expression coefficient error ${worst}`);
});

test('head rotation is recovered, and does not leak into expression', () => {
  const retargeter = new FaceRetargeter(model, { smoothing: 0 });
  const identity = new Float32Array(model.identityDim);

  const turned = { ...STILL, rotation: rotationY(0.35) };
  const solution = retargeter.solve(
    syntheticFrame(identity, new Float32Array(model.expressionDim), turned),
    ASPECT,
  );

  let peak = 0;
  for (const value of solution.expression) peak = Math.max(peak, Math.abs(value));
  console.log(
    `    turned 0.35 rad: solved [${[...solution.headRotation].map((v) => v.toFixed(3)).join(', ')}], ` +
      `peak |expression| ${peak.toFixed(4)}`,
  );

  assert.ok(Math.abs(Math.abs(solution.headRotation[1]) - 0.35) < 0.01, 'yaw not recovered');
  assert.ok(Math.abs(solution.headRotation[0]) < 0.01, 'yaw leaked into pitch');
  assert.ok(Math.abs(solution.headRotation[2]) < 0.01, 'yaw leaked into roll');
  assert.ok(peak < 0.05, `head rotation leaked into expression: ${peak}`);
});

test('smoothing lags towards the answer instead of jumping to it', () => {
  // These frames are generated by the model, so the observations are perfect and
  // the landmark shortfall `regionGain` compensates for does not exist. Leaving it
  // on would measure the compensation rather than the solve.
  const retargeter = new FaceRetargeter(model, { smoothing: 0.5, regionGain: {} });
  const identity = new Float32Array(model.identityDim);
  const truth = mouthExpression();
  const region = manifest.expression.regions['lower_face_region'];
  const frame = syntheticFrame(identity, truth, STILL);

  // The first frame has no history to blend with, so it is taken whole.
  const first = retargeter.solve(frame, ASPECT).expression[region.start];
  assert.ok(Math.abs(first - truth[region.start]) < 0.15, 'first frame should not be damped');

  // A jump back to rest is approached, not snapped to.
  const resting = syntheticFrame(identity, new Float32Array(model.expressionDim), STILL);
  const afterOne = retargeter.solve(resting, ASPECT).expression[region.start];
  assert.ok(
    Math.abs(afterOne) > 0.2 * Math.abs(first),
    `smoothing snapped instead of lagging: ${first} -> ${afterOne}`,
  );

  for (let i = 0; i < 30; i++) retargeter.solve(resting, ASPECT);
  const settled = retargeter.solve(resting, ASPECT).expression[region.start];
  assert.ok(Math.abs(settled) < 0.05, `did not settle back to rest: ${settled}`);
});

test('the region budget confines the solve to the leading components', () => {
  const region = manifest.expression.regions['lower_face_region'];
  const budget = 4;

  // A frame the trailing components alone would explain. They are outside the
  // budget, so the solve must not reach for them -- with sparse noisy
  // landmarks they are what it overfits, and measured on rendered mouth
  // openings that overfit inverted the aperture and read as inflated lips.
  const truth = new Float32Array(model.expressionDim);
  truth[region.start + budget + 2] = 1.5;
  truth[region.start + region.count - 1] = -1.2;

  const retargeter = new FaceRetargeter(model, { smoothing: 0 });
  const solution = retargeter.solve(syntheticFrame(new Float32Array(model.identityDim), truth, STILL), ASPECT);

  let leaked = 0;
  for (let i = budget; i < region.count; i++) {
    leaked = Math.max(leaked, Math.abs(solution.expression[region.start + i]));
  }
  console.log(`    coefficients outside the budget: peak ${leaked.toFixed(6)}`);
  assert.equal(leaked, 0, `budgeted-out components moved by ${leaked}`);

  // The eye regions are left unbudgeted, so they must still be solvable.
  const eye = manifest.expression.regions['left_eye_region'];
  const eyeTruth = new Float32Array(model.expressionDim);
  eyeTruth[eye.start + eye.count - 1] = 1.0;
  const eyeSolution = new FaceRetargeter(model, { smoothing: 0 }).solve(
    syntheticFrame(new Float32Array(model.identityDim), eyeTruth, STILL), ASPECT,
  );
  assert.ok(
    Math.abs(eyeSolution.expression[eye.start + eye.count - 1]) > 0.1,
    'a trailing eye component should still be solvable',
  );
});
