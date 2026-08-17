/**
 * Checks the TypeScript forward pass against the Python reference.
 *
 * `tools/reference_model.py` is verified bit-exact against `gnm.shape`, and
 * `tools/dump_test_vectors.py` freezes its output. Matching those frozen
 * vectors here is what carries that guarantee across the language boundary.
 *
 * Regenerate the inputs first, from the repository root:
 *   python -m web_puppet.tools.export_assets --output_dir web_puppet/public/assets
 *   python -m web_puppet.tools.dump_test_vectors
 *
 * Then, from web_puppet/:
 *   npm test
 */

import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import test from 'node:test';

import { GnmModel, type Manifest } from './gnm.ts';

const here = dirname(fileURLToPath(import.meta.url));
const assetDir = join(here, '..', 'public', 'assets');
const goldenDir = join(here, '..', 'testdata');

// Both sides evaluate in float32 from identical float16 inputs, so they should
// agree to within accumulation-order noise -- not to within the quantization
// error, which is common to both and cancels. A micrometre is generous for
// that and still four orders of magnitude below the truncation error the
// assets already carry.
const TOLERANCE_MM = 1e-3;

async function loadBuffer(path: string): Promise<ArrayBuffer> {
  const data = await readFile(path);
  return data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) as ArrayBuffer;
}

interface GoldenCase {
  name: string;
  identity: number[];
  expression: number[];
  rotations: number[][];
  translation: number[];
  byteOffset: number;
  byteLength: number;
}

const manifest = JSON.parse(
  await readFile(join(assetDir, 'gnm_head.json'), 'utf8'),
) as Manifest;
const model = new GnmModel(manifest, await loadBuffer(join(assetDir, 'gnm_head.bin')));

const golden = JSON.parse(await readFile(join(goldenDir, 'golden.json'), 'utf8')) as {
  vertexCount: number;
  identityDim: number;
  expressionDim: number;
  cases: GoldenCase[];
};
const goldenBuffer = await loadBuffer(join(goldenDir, 'golden.bin'));

test('asset dimensions match the golden dump', () => {
  assert.equal(model.vertexCount, golden.vertexCount);
  assert.equal(model.identityDim, golden.identityDim);
  assert.equal(model.expressionDim, golden.expressionDim);
});

for (const testCase of golden.cases) {
  test(`matches the Python reference: ${testCase.name}`, () => {
    const expected = new Float32Array(
      goldenBuffer,
      testCase.byteOffset,
      testCase.byteLength / 4,
    );

    const actual = model.evaluate(
      testCase.identity,
      testCase.expression,
      testCase.rotations.flat(),
      testCase.translation,
    );

    assert.equal(actual.length, expected.length);

    let worst = 0;
    let sumSquares = 0;
    for (let v = 0; v < model.vertexCount; v++) {
      const dx = actual[v * 3] - expected[v * 3];
      const dy = actual[v * 3 + 1] - expected[v * 3 + 1];
      const dz = actual[v * 3 + 2] - expected[v * 3 + 2];
      const distance = Math.hypot(dx, dy, dz) * 1000;
      worst = Math.max(worst, distance);
      sumSquares += distance * distance;
    }
    const rms = Math.sqrt(sumSquares / model.vertexCount);
    console.log(`    ${testCase.name}: max ${worst.toExponential(2)} mm, rms ${rms.toExponential(2)} mm`);

    assert.ok(worst < TOLERANCE_MM, `max error ${worst} mm exceeds ${TOLERANCE_MM} mm`);
  });
}

test('the posed case actually moves the mesh', () => {
  // Guards against every case agreeing because skinning silently no-ops: if
  // the transforms collapsed to identity, `posed` would equal `template` and
  // both would still match their goldens.
  const template = new Float32Array(goldenBuffer, golden.cases[0].byteOffset, golden.cases[0].byteLength / 4);
  const posed = golden.cases.find((c) => c.name === 'posed')!;
  const posedVertices = new Float32Array(goldenBuffer, posed.byteOffset, posed.byteLength / 4);

  let worst = 0;
  for (let i = 0; i < template.length; i++) {
    worst = Math.max(worst, Math.abs(posedVertices[i] - template[i]));
  }
  assert.ok(worst > 0.005, `posing moved at most ${worst * 1000} mm`);
});

test('component grouping partitions the index buffer', () => {
  const { indices, groups } = model.groupTrianglesByComponent();
  assert.equal(indices.length, model.triangles.length);

  const total = groups.reduce((sum, g) => sum + g.count, 0);
  assert.equal(total, model.triangles.length);

  let expectedStart = 0;
  for (const group of groups) {
    assert.equal(group.start, expectedStart);
    expectedStart += group.count;
  }

  // Every triangle in a group must belong to that group's component.
  for (const group of groups) {
    for (let i = group.start; i < group.start + group.count; i++) {
      assert.equal(model.componentIds[indices[i]], group.component);
    }
  }
});

test('normals are unit length and point outward at the nose', () => {
  const vertices = model.evaluate(
    new Float32Array(model.identityDim),
    new Float32Array(model.expressionDim),
    new Float32Array(model.jointCount * 3),
    new Float32Array(3),
  );
  const normals = model.computeNormals(vertices);

  for (let i = 0; i < normals.length; i += 3) {
    const length = Math.hypot(normals[i], normals[i + 1], normals[i + 2]);
    assert.ok(Math.abs(length - 1) < 1e-4, `normal ${i / 3} has length ${length}`);
  }

  // The frontmost skin vertex is on the nose or lips; either way its normal
  // must face the camera, which catches a flipped winding order.
  let frontmost = 0;
  for (let v = 0; v < model.vertexCount; v++) {
    if (model.componentIds[v] === 0 && vertices[v * 3 + 2] > vertices[frontmost * 3 + 2]) {
      frontmost = v;
    }
  }
  assert.ok(normals[frontmost * 3 + 2] > 0.5, `frontmost normal z is ${normals[frontmost * 3 + 2]}`);
});

test('the corrective basis loads and moves only the cheeks', () => {
  assert.ok(model.correctiveDim > 0, 'no correctives were exported');
  assert.equal(model.correctiveBlendshapes.length, model.correctiveDim);
  assert.deepEqual(model.correctiveBlendshapes, ['cheekPuff']);

  const zero = new Float32Array(model.expressionDim);
  const identity = new Float32Array(model.identityDim);
  const neutral = Float32Array.from(model.bindPose(identity, zero));
  const puffed = model.bindPose(identity, zero, Float32Array.from([1]));

  let moved = 0;
  let peak = 0;
  let sumY = 0;
  for (let v = 0; v < model.vertexCount; v++) {
    const d = Math.hypot(
      puffed[v * 3] - neutral[v * 3],
      puffed[v * 3 + 1] - neutral[v * 3 + 1],
      puffed[v * 3 + 2] - neutral[v * 3 + 2],
    ) * 1000;
    if (d > 0.01) { moved++; sumY += neutral[v * 3 + 1]; }
    peak = Math.max(peak, d);
  }
  console.log(`    corrective: ${moved} vertices move, peak ${peak.toFixed(1)} mm`);

  // A puff, not a global scale: a bounded patch at roughly the amplitude a
  // real cheek puff spans.
  assert.ok(moved > 200 && moved < model.vertexCount * 0.2, `${moved} vertices moved`);
  assert.ok(peak > 8 && peak < 16, `peak displacement ${peak} mm`);

  // Both cheeks, so the mean of the moved vertices sits near the midline.
  let sumX = 0;
  let left = 0;
  for (let v = 0; v < model.vertexCount; v++) {
    const d = Math.abs(puffed[v * 3] - neutral[v * 3])
      + Math.abs(puffed[v * 3 + 1] - neutral[v * 3 + 1])
      + Math.abs(puffed[v * 3 + 2] - neutral[v * 3 + 2]);
    if (d * 1000 > 0.01) { sumX += neutral[v * 3]; if (neutral[v * 3] > 0) left++; }
  }
  const balance = left / moved;
  console.log(`    sides: ${(100 * balance).toFixed(0)}% on +x, centroid x ${(1000 * sumX / moved).toFixed(1)} mm`);
  assert.ok(balance > 0.35 && balance < 0.65, `lopsided puff: ${balance}`);

  // Zero weight must be a no-op, or the neutral face is quietly wrong.
  const unweighted = model.bindPose(identity, zero, Float32Array.from([0]));
  let worst = 0;
  for (let i = 0; i < neutral.length; i++) {
    worst = Math.max(worst, Math.abs(unweighted[i] - neutral[i]));
  }
  assert.equal(worst, 0, `zero weight moved the mesh by ${worst}`);
});
