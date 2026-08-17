/**
 * Copies MediaPipe's runtime files into public/ so the page is self-contained.
 *
 * The usual guidance is to point FilesetResolver at a CDN. Vendoring instead
 * keeps the app working offline, keeps the WASM version pinned to the npm
 * package that the TypeScript types came from -- a mismatch there fails at
 * runtime, not at build -- and means no third-party host sees a request from
 * a page that is about to open a webcam.
 *
 * Run from web_puppet/:  npm run sync-assets
 */

import { cp, mkdir, stat } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');

const WASM_SOURCE = join(root, 'node_modules', '@mediapipe', 'tasks-vision', 'wasm');
const WASM_TARGET = join(root, 'public', 'mediapipe', 'wasm');

// The face landmarker bundle, already fetched by the earlier prototype. If it
// is ever missing it can be re-downloaded from:
// https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
const MODEL_SOURCE = join(root, '..', 'webcam_puppet', 'assets', 'face_landmarker.task');
const MODEL_TARGET = join(root, 'public', 'assets', 'face_landmarker.task');

async function exists(path) {
  try {
    await stat(path);
    return true;
  } catch {
    return false;
  }
}

async function main() {
  if (!(await exists(WASM_SOURCE))) {
    throw new Error(`Missing ${WASM_SOURCE}. Run npm install first.`);
  }
  await mkdir(WASM_TARGET, { recursive: true });
  await cp(WASM_SOURCE, WASM_TARGET, { recursive: true });
  console.log(`wasm    -> ${WASM_TARGET}`);

  if (!(await exists(MODEL_SOURCE))) {
    throw new Error(
      `Missing ${MODEL_SOURCE}. Download face_landmarker.task from ` +
        'https://storage.googleapis.com/mediapipe-models/face_landmarker/' +
        'face_landmarker/float16/1/face_landmarker.task',
    );
  }
  await mkdir(dirname(MODEL_TARGET), { recursive: true });
  await cp(MODEL_SOURCE, MODEL_TARGET);
  const { size } = await stat(MODEL_TARGET);
  console.log(`model   -> ${MODEL_TARGET} (${(size / 1e6).toFixed(1)} MB)`);
}

await main();
