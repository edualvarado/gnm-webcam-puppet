/**
 * Checks the identity fit against faces whose answer is known in advance.
 *
 * A real webcam frame has no ground truth -- that is the whole difficulty of
 * M4b -- so the model generates its own: pick coefficients, evaluate the
 * correspondence vertices, push them through a known similarity transform and
 * back into MediaPipe's normalized convention. Recovering the coefficients
 * from that is a round trip the solver either closes or does not.
 *
 * Run from web_puppet/:
 *   npm test
 */

import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import test from 'node:test';

import { GnmModel, type Manifest } from './gnm.ts';
import {
  applySimilarity,
  fitIdentity,
  fitSimilarity,
  type Landmark,
  type SimilarityTransform,
} from './fit.ts';

const here = dirname(fileURLToPath(import.meta.url));
const assetDir = join(here, '..', 'public', 'assets');

/** A 1280x720 camera, matching what `FacePip` requests. */
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

/** A rotation about Y, as the row-major 3x3 the transform type wants. */
function rotationY(angle: number): Float64Array {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return Float64Array.from([c, 0, s, 0, 1, 0, -s, 0, c]);
}

/**
 * Builds the landmark frame a face with these coefficients would produce.
 *
 * This is the inverse of what the solver does: evaluate the shape, place it in
 * the camera's frame with a known transform, and express the result the way
 * MediaPipe would report it.
 *
 * Landmarks start from the reference cloud, not from the vertices they map to,
 * because that is where the landmarker actually puts them on a neutral face --
 * the two sit 11 mm rms apart in z. Modelling that offset here is what makes
 * the round trip a test of the solver rather than of an assumption the solver
 * does not make.
 */
function syntheticFrame(
  identity: ArrayLike<number>,
  transform: SimilarityTransform,
): Landmark[] {
  const stride = model.vertexCount * 3;

  const points = new Float64Array(correspondence.landmarks.length * 3);
  for (let i = 0; i < correspondence.landmarks.length; i++) {
    const vertex = correspondence.vertices[i];
    for (let axis = 0; axis < 3; axis++) {
      let value = correspondence.reference[i * 3 + axis];
      for (let k = 0; k < model.identityDim; k++) {
        value += identity[k] * model.identityBasis[k * stride + vertex * 3 + axis];
      }
      points[i * 3 + axis] = value;
    }
  }

  const placed = applySimilarity(points, transform, new Float64Array(points.length));

  const highest = Math.max(...correspondence.landmarks) + 1;
  const landmarks: Landmark[] = Array.from({ length: highest }, () => ({ x: 0, y: 0, z: 0 }));
  for (let i = 0; i < correspondence.landmarks.length; i++) {
    // Undo `landmarksToModelAxes`, so the solver's first step reproduces
    // `placed` exactly.
    landmarks[correspondence.landmarks[i]] = {
      x: placed[i * 3] / ASPECT,
      y: -placed[i * 3 + 1],
      z: -placed[i * 3 + 2] / ASPECT,
    };
  }
  return landmarks;
}

/** The transform standing in for "where the head was when the shutter fired". */
const PLACEMENT: SimilarityTransform = {
  scale: 0.42,
  rotation: rotationY(0.15),
  translation: Float64Array.from([0.05, -0.02, 0.3]),
};

test('the similarity fit recovers a known transform', () => {
  const source = new Float64Array(30);
  for (let i = 0; i < source.length; i++) source[i] = Math.sin(i * 1.7) * 0.1;

  const expected: SimilarityTransform = {
    scale: 1.8,
    rotation: rotationY(0.4),
    translation: Float64Array.from([0.3, -0.1, 0.05]),
  };
  const target = applySimilarity(source, expected, new Float64Array(source.length));

  const actual = fitSimilarity(source, target);
  assert.ok(
    Math.abs(actual.scale - expected.scale) < 1e-9,
    `scale ${actual.scale} != ${expected.scale}`,
  );
  for (let i = 0; i < 9; i++) {
    assert.ok(
      Math.abs(actual.rotation[i] - expected.rotation[i]) < 1e-9,
      `rotation[${i}] ${actual.rotation[i]} != ${expected.rotation[i]}`,
    );
  }
  for (let i = 0; i < 3; i++) {
    assert.ok(
      Math.abs(actual.translation[i] - expected.translation[i]) < 1e-9,
      `translation[${i}] ${actual.translation[i]} != ${expected.translation[i]}`,
    );
  }
});

test('the similarity fit is not fooled into a reflection', () => {
  // A cloud that is nearly planar is where an SVD-based fit can return a
  // determinant -1 "rotation"; the quaternion form cannot represent one.
  const source = new Float64Array(36);
  for (let i = 0; i < 12; i++) {
    source[i * 3] = Math.cos(i);
    source[i * 3 + 1] = Math.sin(i);
    source[i * 3 + 2] = 1e-6 * i;
  }
  const target = applySimilarity(
    source,
    { scale: 1, rotation: rotationY(2.9), translation: Float64Array.from([0, 0, 0]) },
    new Float64Array(source.length),
  );

  const { rotation: r } = fitSimilarity(source, target);
  const determinant =
    r[0] * (r[4] * r[8] - r[5] * r[7]) -
    r[1] * (r[3] * r[8] - r[5] * r[6]) +
    r[2] * (r[3] * r[7] - r[4] * r[6]);
  assert.ok(Math.abs(determinant - 1) < 1e-9, `determinant is ${determinant}`);
});

test('a template face fits as the template', () => {
  const landmarks = syntheticFrame(new Float32Array(model.identityDim), PLACEMENT);
  const fit = fitIdentity(model, landmarks, ASPECT);

  console.log(
    `    template: ${fit.points} points, rms ${fit.rmsAfter.toFixed(4)} mm, peak |c| ${fit.peak.toFixed(4)}`,
  );
  assert.equal(fit.points, 166);
  assert.ok(fit.rmsAfter < 0.05, `rms ${fit.rmsAfter} mm on an exact template face`);
  assert.ok(fit.peak < 0.05, `peak coefficient ${fit.peak} on an exact template face`);
});

/** How far one identity's face sits from another's, over the whole mesh. */
function meshDistance(a: ArrayLike<number>, b: ArrayLike<number>): { rms: number; max: number } {
  const zero = new Float32Array(model.expressionDim);
  const first = Float32Array.from(model.bindPose(a, zero));
  const second = model.bindPose(b, zero);

  let sum = 0;
  let worst = 0;
  for (let v = 0; v < model.vertexCount; v++) {
    const distance = Math.hypot(
      second[v * 3] - first[v * 3],
      second[v * 3 + 1] - first[v * 3 + 1],
      second[v * 3 + 2] - first[v * 3 + 2],
    ) * 1000;
    sum += distance * distance;
    worst = Math.max(worst, distance);
  }
  return { rms: Math.sqrt(sum / model.vertexCount), max: worst };
}

test('a known face is recovered as a face, if not as coefficients', () => {
  const truth = new Float32Array(model.identityDim);
  truth[0] = 1.2;
  truth[1] = -0.8;
  truth[2] = 0.6;
  truth[3] = -0.4;

  // Ridge is what keeps a real, noisy frame plausible; here the data is exact,
  // so it is turned down to measure the solver rather than the prior.
  const fit = fitIdentity(model, syntheticFrame(truth, PLACEMENT), ASPECT, { ridge: 1e-5 });

  const recovered = meshDistance(truth, fit.identity);
  const doingNothing = meshDistance(truth, new Float32Array(model.identityDim));
  let worstCoefficient = 0;
  for (let k = 0; k < 4; k++) {
    worstCoefficient = Math.max(worstCoefficient, Math.abs(fit.identity[k] - truth[k]));
  }
  console.log(
    `    landmark rms ${fit.rmsBefore.toFixed(2)} -> ${fit.rmsAfter.toFixed(3)} mm | ` +
      `face vs truth ${recovered.rms.toFixed(2)} mm rms (template would be ` +
      `${doingNothing.rms.toFixed(2)}) | worst coefficient error ${worstCoefficient.toFixed(2)}`,
  );

  assert.ok(fit.rmsAfter < 0.25, `landmark rms ${fit.rmsAfter} mm on an exact face`);

  // The coefficients are deliberately *not* asserted. 166 sparse landmarks
  // leave a large near-null space in a 64-component basis, so a different
  // coefficient vector can produce the same face to well under a millimetre at
  // the fitted points. What has to hold is that the face is right, and that
  // fitting beats not fitting by a wide margin.
  assert.ok(
    recovered.rms < doingNothing.rms * 0.5,
    `fitted face is ${recovered.rms} mm from truth, template is ${doingNothing.rms} mm`,
  );
});

test('fitting improves on the template it starts from', () => {
  const truth = new Float32Array(model.identityDim);
  truth[0] = 1.5;
  truth[4] = 1.0;
  truth[7] = -1.1;

  const fit = fitIdentity(model, syntheticFrame(truth, PLACEMENT), ASPECT);
  console.log(`    rms ${fit.rmsBefore.toFixed(2)} -> ${fit.rmsAfter.toFixed(2)} mm`);
  assert.ok(
    fit.rmsAfter < fit.rmsBefore * 0.5,
    `fitting moved rms from ${fit.rmsBefore} to ${fit.rmsAfter} mm`,
  );
});

test('the fit is invariant to where the head sits in frame', () => {
  const truth = new Float32Array(model.identityDim);
  truth[0] = 0.9;
  truth[2] = -1.1;

  const near = fitIdentity(model, syntheticFrame(truth, PLACEMENT), ASPECT, { ridge: 1e-5 });
  const far = fitIdentity(
    model,
    syntheticFrame(truth, {
      scale: 0.18,
      rotation: rotationY(-0.35),
      translation: Float64Array.from([-0.2, 0.15, 1.1]),
    }),
    ASPECT,
    { ridge: 1e-5 },
  );

  let worst = 0;
  for (let k = 0; k < model.identityDim; k++) {
    worst = Math.max(worst, Math.abs(near.identity[k] - far.identity[k]));
  }
  console.log(`    worst coefficient difference across placements ${worst.toFixed(4)}`);
  assert.ok(worst < 0.05, `placement changed a coefficient by ${worst}`);
});

test('ridge shrinks the fit towards the template', () => {
  const truth = new Float32Array(model.identityDim);
  truth[0] = 2.0;
  const landmarks = syntheticFrame(truth, PLACEMENT);

  const loose = fitIdentity(model, landmarks, ASPECT, { ridge: 1e-5 });
  const tight = fitIdentity(model, landmarks, ASPECT, { ridge: 100 });
  assert.ok(
    tight.peak < loose.peak * 0.5,
    `ridge did not shrink: peak ${loose.peak} -> ${tight.peak}`,
  );
});

test('coefficients stay inside the clamp', () => {
  // An absurd face, to check the limit binds rather than the solver diverging.
  const truth = new Float32Array(model.identityDim);
  for (let k = 0; k < model.identityDim; k++) truth[k] = (k % 2 ? -1 : 1) * 6;

  const fit = fitIdentity(model, syntheticFrame(truth, PLACEMENT), ASPECT, { limit: 3 });
  for (let k = 0; k < model.identityDim; k++) {
    assert.ok(Math.abs(fit.identity[k]) <= 3 + 1e-9, `coefficient ${k} is ${fit.identity[k]}`);
  }
  assert.ok(Number.isFinite(fit.rmsAfter), 'rms is not finite');
});

test('a frame shorter than the correspondence is rejected', () => {
  assert.throws(
    () => fitIdentity(model, [{ x: 0, y: 0, z: 0 }], ASPECT),
    /correspondence indexes landmark/,
  );
});
