/**
 * M1 entry point: a static GNM head with sliders, no camera involved.
 *
 * The point of this page is to judge the look and to prove the browser
 * forward pass drives the mesh correctly. Tracking arrives in M3; until then
 * the sliders stand in for whatever will eventually write these coefficients.
 */

import { fitIdentity } from './fit.ts';
import { loadGnmModel, type GnmModel } from './gnm.ts';
import { Viewer } from './look.ts';
import { FacePip } from './pip.ts';
import { FaceRetargeter } from './retarget.ts';

/** How many components of each region get a slider. */
const SLIDER_BUDGET: Record<string, number> = {
  lower_face_region: 6,
  left_eye_region: 3,
  right_eye_region: 3,
};
const IDENTITY_SLIDERS = 8;
// Smoothing on the corrective weights. Heavier than the expression solve's,
// because a blendshape score is a classifier output rather than a fit and
// flickers between adjacent frames.
const CORRECTIVE_ALPHA = 0.35;

interface State {
  identity: Float32Array;
  expression: Float32Array;
  rotations: Float32Array;
  translation: Float32Array;
  /** Weights for the authored shapes GNM cannot make. */
  correctives: Float32Array;
}

function element<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function addSlider(
  parent: HTMLElement,
  label: string,
  min: number,
  max: number,
  value: number,
  onInput: (value: number) => void,
): HTMLInputElement {
  const row = element('div', 'row');
  const name = element('label', 'name', label);
  const readout = element('span', 'value', value.toFixed(2));
  const input = element('input');
  input.type = 'range';
  input.min = String(min);
  input.max = String(max);
  input.step = '0.01';
  input.value = String(value);
  input.addEventListener('input', () => {
    const next = Number(input.value);
    readout.textContent = next.toFixed(2);
    onInput(next);
  });
  row.append(name, input, readout);
  parent.append(row);
  return input;
}

/**
 * Compares the GPU's posed vertices against the CPU forward pass.
 *
 * The CPU path is verified against Python in `gnm.test.ts`, so agreeing with
 * it here extends that chain all the way to the shader. There is no headless
 * WebGL in the Node test, so this runs in the page instead and reports into
 * the HUD.
 */
function verify(model: GnmModel, viewer: Viewer, state: State): string {
  // Drive the GPU from this exact state rather than trusting whatever the
  // frame loop last rendered. Without this the check silently compares the
  // CPU's current answer against the GPU's previous one, and reports a large
  // error that is an artefact of the harness rather than of the shader.
  viewer.update(state.identity, state.expression, state.rotations, state.translation, state.correctives);

  const expected = model.evaluate(
    state.identity, state.expression, state.rotations, state.translation, undefined,
    state.correctives,
  );
  const actual = viewer.gpu.readPositions();

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
  return `gpu vs cpu — max ${worst.toExponential(2)} mm, rms ${rms.toExponential(2)} mm`;
}

function buildUi(
  model: GnmModel,
  viewer: Viewer,
  state: State,
  invalidate: () => void,
  onVerify: () => string,
): void {
  const panel = document.querySelector<HTMLElement>('#panel')!;
  const sliders: { input: HTMLInputElement; reset: number }[] = [];

  const register = (input: HTMLInputElement, reset: number) => {
    sliders.push({ input, reset });
  };

  // --- Head pose -----------------------------------------------------------
  panel.append(element('h2', undefined, 'Head pose'));
  const poseGroup = element('div', 'group');
  panel.append(poseGroup);
  const HEAD_JOINT = model.manifest.model.jointNames.indexOf('head');
  (['pitch', 'yaw', 'roll'] as const).forEach((axis, index) => {
    const input = addSlider(poseGroup, axis, -0.7, 0.7, 0, (value) => {
      state.rotations[HEAD_JOINT * 3 + index] = value;
      invalidate();
    });
    register(input, 0);
  });

  // --- Correctives ---------------------------------------------------------
  // Kept in their own group, above expression and labelled as not-GNM, because
  // that distinction is the whole point: these are authored shapes the model's
  // basis cannot reach, not model output.
  const correctives = model.manifest.correctives;
  if (correctives && model.correctiveDim > 0) {
    panel.append(element('h2', undefined, 'Correctives (not GNM)'));
    const group = element('div', 'group');
    panel.append(group);
    for (let i = 0; i < model.correctiveDim; i++) {
      const label = `${correctives.names[i]} ← ${correctives.blendshapes[i]}`;
      const input = addSlider(group, label, 0, 1, 0, (value) => {
        state.correctives[i] = value;
        invalidate();
      });
      register(input, 0);
    }
  }

  // --- Expression ----------------------------------------------------------
  panel.append(element('h2', undefined, 'Expression'));
  const expressionGroup = element('div', 'group');
  panel.append(expressionGroup);
  for (const [region, budget] of Object.entries(SLIDER_BUDGET)) {
    const span = model.manifest.expression.regions[region];
    if (!span) continue;
    for (let i = 0; i < Math.min(budget, span.count); i++) {
      const index = span.start + i;
      const label = model.manifest.expression.names[index];
      const input = addSlider(expressionGroup, label, -3, 3, 0, (value) => {
        state.expression[index] = value;
        invalidate();
      });
      register(input, 0);
    }
  }

  // --- Identity ------------------------------------------------------------
  panel.append(element('h2', undefined, 'Identity'));
  const identityGroup = element('div', 'group');
  panel.append(identityGroup);
  const identityInputs: HTMLInputElement[] = [];
  for (let i = 0; i < Math.min(IDENTITY_SLIDERS, model.identityDim); i++) {
    const label = model.manifest.identity.names[i];
    const input = addSlider(identityGroup, label, -3, 3, 0, (value) => {
      state.identity[i] = value;
      invalidate();
    });
    identityInputs.push(input);
    register(input, 0);
  }

  // --- Look ----------------------------------------------------------------
  panel.append(element('h2', undefined, 'Look'));
  const lookGroup = element('div', 'group');
  panel.append(lookGroup);
  addSlider(lookGroup, 'wireframe', 0, 1, 0.35, (v) => viewer.setWireframeOpacity(v));
  addSlider(lookGroup, 'bloom', 0, 2, 0.38, (v) => viewer.setBloomStrength(v));

  // --- Buttons -------------------------------------------------------------
  const buttons = element('div', 'buttons');
  panel.append(buttons);

  const randomize = element('button', undefined, 'Random identity');
  randomize.addEventListener('click', () => {
    for (let i = 0; i < model.identityDim; i++) {
      // Damp the tail so a random draw reads as a plausible face rather than
      // as noise: later components carry finer detail and go wrong faster.
      state.identity[i] = (Math.random() * 2 - 1) * (i < 16 ? 1.4 : 0.5);
    }
    identityInputs.forEach((input, i) => {
      input.value = String(state.identity[i]);
      input.dispatchEvent(new Event('input'));
    });
    invalidate();
  });

  const reset = element('button', undefined, 'Reset');
  reset.addEventListener('click', () => {
    state.identity.fill(0);
    state.expression.fill(0);
    state.rotations.fill(0);
    state.correctives.fill(0);
    for (const slider of sliders) {
      slider.input.value = String(slider.reset);
      slider.input.dispatchEvent(new Event('input'));
    }
    invalidate();
  });

  const toggleWire = element('button', undefined, 'Toggle wireframe');
  let wireVisible = true;
  toggleWire.addEventListener('click', () => {
    wireVisible = !wireVisible;
    viewer.setWireframeVisible(wireVisible);
  });

  const check = element('button', undefined, 'Verify vs CPU');
  check.addEventListener('click', () => {
    document.querySelector<HTMLElement>('#verify')!.textContent = onVerify();
  });

  buttons.append(randomize, reset, toggleWire, check);

  // --- Camera --------------------------------------------------------------
  // The tracker drives identity, and nothing else. Expression and pose stay on
  // the sliders until M4c, so a face that looks wrong after a fit can only be
  // the fit.
  panel.append(element('h2', undefined, 'Camera'));
  const cameraGroup = element('div', 'group');
  panel.append(cameraGroup);

  const pipCanvas = document.querySelector<HTMLCanvasElement>('#pip')!;
  const pipStatus = document.querySelector<HTMLElement>('#pipStatus')!;
  const pip = new FacePip(pipCanvas, { size: 480 });

  const cameraButtons = element('div', 'buttons');
  panel.append(cameraButtons);

  const camera = element('button', undefined, 'Start camera');
  camera.addEventListener('click', async () => {
    if (pip.isRunning) {
      pip.stop();
      pipCanvas.classList.remove('live');
      pipStatus.textContent = '';
      camera.textContent = 'Start camera';
      return;
    }
    camera.disabled = true;
    camera.textContent = 'Starting…';
    try {
      await pip.start();
      pipCanvas.classList.add('live');
      camera.textContent = 'Stop camera';
    } catch (error) {
      pipStatus.textContent = String(error);
      camera.textContent = 'Start camera';
    } finally {
      camera.disabled = false;
    }
  });

  const mirror = element('button', undefined, 'Mirror');
  mirror.addEventListener('click', () => { pip.mirror = !pip.mirror; });

  cameraButtons.append(camera, mirror);

  // --- Identity fit ---------------------------------------------------------
  // M4b: the first thing tracking actually drives. Identity only, from a
  // single frame, on the skull-fixed landmarks -- expression and pose are
  // still slider-driven, which is what keeps a bad fit diagnosable.
  const fitButtons = element('div', 'buttons');
  panel.append(fitButtons);

  const fit = element('button', undefined, 'Fit identity');
  fit.addEventListener('click', () => {
    const verify = document.querySelector<HTMLElement>('#verify')!;
    const frame = pip.latest;
    if (!frame) {
      verify.textContent = pip.isRunning
        ? 'fit needs a detected face'
        : 'fit needs the camera running';
      return;
    }

    const started = performance.now();
    const result = fitIdentity(model, frame.landmarks, pip.aspect);
    const elapsed = performance.now() - started;

    state.identity.set(result.identity);
    identityInputs.forEach((input, i) => {
      input.value = String(state.identity[i]);
      input.dispatchEvent(new Event('input'));
    });
    // Expression is measured as displacement from this subject's neutral face,
    // so the retargeter has to be told when that face changes.
    retargeter.setIdentity(state.identity);
    invalidate();

    verify.textContent =
      `fit ${result.components} components from ${result.points} points · ` +
      `landmark rms ${result.rmsBefore.toFixed(1)} -> ${result.rmsAfter.toFixed(1)} mm · ` +
      `peak |c| ${result.peak.toFixed(2)} · ${elapsed.toFixed(0)} ms`;
  });
  fitButtons.append(fit);

  // --- M4c: the head follows the face ---------------------------------------
  const retargeter = new FaceRetargeter(model);
  const HEAD = model.manifest.model.jointNames.indexOf('head');

  const drive = element('button', undefined, 'Drive head');
  let driving = false;

  const stopDriving = () => {
    driving = false;
    drive.textContent = 'Drive head';
    // Hand the face back to the sliders where it was left, rather than frozen
    // mid-word.
    state.expression.fill(0);
    state.rotations.fill(0);
    state.correctives.fill(0);
    for (const slider of sliders) {
      slider.input.value = String(slider.reset);
      slider.input.dispatchEvent(new Event('input'));
    }
    invalidate();
  };

  drive.addEventListener('click', () => {
    if (driving) {
      stopDriving();
      return;
    }
    if (!pip.isRunning) {
      document.querySelector<HTMLElement>('#verify')!.textContent =
        'driving needs the camera running';
      return;
    }
    driving = true;
    retargeter.reset();
    drive.textContent = 'Stop driving';
  });
  fitButtons.append(drive);

  // The retarget runs on its own timer rather than in the render loop: the
  // camera produces frames at 30 fps and the page renders at 60, so solving
  // per rendered frame would do half its work on landmarks that had not
  // changed.
  setInterval(() => {
    if (!driving) return;
    if (!pip.isRunning) {
      stopDriving();
      return;
    }
    const frame = pip.latest;
    if (!frame) return;

    const solution = retargeter.solve(frame.landmarks, pip.aspect);
    state.expression.set(solution.expression);
    if (HEAD >= 0) state.rotations.set(solution.headRotation, HEAD * 3);

    // Correctives come straight from the blendshape classifier rather than
    // from the landmark solve. They have to: the shapes they drive are not in
    // the expression basis, so there is nothing for a landmark residual to
    // project onto. A cheek puff is also nearly invisible head-on -- it lives
    // in the silhouette -- which is the other reason the landmarks miss it.
    const driven = model.correctiveBlendshapes;
    for (let i = 0; i < driven.length; i++) {
      const score = frame.blendshapes.get(driven[i]) ?? 0;
      state.correctives[i] += (score - state.correctives[i]) * CORRECTIVE_ALPHA;
    }
    invalidate();

    document.querySelector<HTMLElement>('#verify')!.textContent =
      `driving · landmark rms ${solution.rms.toFixed(1)} mm · ` +
      `head ${solution.headRotation[0].toFixed(2)}, ` +
      `${solution.headRotation[1].toFixed(2)}, ${solution.headRotation[2].toFixed(2)} rad`;
  }, 33);

  // --- Correspondence ------------------------------------------------------
  // The same 473 points, drawn on the camera image and on the mesh in
  // matching colours. Hue follows horizontal position and lightness follows
  // vertical, so the colour is a readable function of where a point sits on a
  // face and the two views can be matched by eye.
  if (model.correspondence && viewer.hasCorrespondence) {
    const mapping = model.correspondence;
    const correspondenceButtons = element('div', 'buttons');
    panel.append(correspondenceButtons);

    const toggle = element(
      'button', undefined, `Correspondence (${mapping.landmarks.length})`,
    );
    let showing = false;
    toggle.addEventListener('click', () => {
      showing = !showing;
      viewer.setCorrespondenceVisible(showing);
      pip.correspondence = showing
        ? { landmarks: mapping.landmarks, colors: mapping.colors }
        : null;
      toggle.textContent = showing
        ? 'Hide correspondence'
        : `Correspondence (${mapping.landmarks.length})`;
    });
    correspondenceButtons.append(toggle);
  }

  // The head is slider-driven, so its own loop only redraws when something
  // changes. The tracker readout has to tick regardless.
  setInterval(() => {
    if (!pip.isRunning) return;
    const frame = pip.latest;
    if (!frame) {
      pipStatus.textContent = `no face · ${pip.fps.toFixed(0)} fps`;
      return;
    }
    const gaze = frame.gaze!;
    const blink = Math.max(
      frame.blendshapes.get('eyeBlinkLeft') ?? 0,
      frame.blendshapes.get('eyeBlinkRight') ?? 0,
    );
    pipStatus.textContent =
      `${pip.fps.toFixed(0)} fps · gaze ${gaze.left.direction.x.toFixed(2)}, ` +
      `${gaze.left.direction.y.toFixed(2)} · blink ${blink.toFixed(2)}`;
  }, 150);
}

async function main(): Promise<void> {
  const status = document.querySelector<HTMLElement>('#status')!;
  const canvas = document.querySelector<HTMLCanvasElement>('#view')!;

  status.textContent = 'Loading model (17 MB)...';
  const model = await loadGnmModel('assets');

  const state: State = {
    identity: new Float32Array(model.identityDim),
    expression: new Float32Array(model.expressionDim),
    rotations: new Float32Array(model.jointCount * 3),
    translation: new Float32Array(3),
    correctives: new Float32Array(model.correctiveDim),
  };

  const vertices = new Float32Array(model.vertexCount * 3);

  const viewer = new Viewer(canvas, model);

  let dirty = true;
  const invalidate = () => { dirty = true; };

  // Frame from the CPU evaluation of the neutral pose: the camera needs real
  // bounds before the first GPU pass has run.
  model.evaluate(
    state.identity, state.expression, state.rotations, state.translation, vertices,
    state.correctives,
  );
  viewer.update(state.identity, state.expression, state.rotations, state.translation, state.correctives);
  viewer.frame(vertices);

  buildUi(model, viewer, state, invalidate, () => verify(model, viewer, state));

  const resize = () => {
    const rect = canvas.parentElement!.getBoundingClientRect();
    viewer.setSize(rect.width, rect.height);
  };
  window.addEventListener('resize', resize);
  resize();

  status.textContent =
    `${model.vertexCount.toLocaleString()} verts · ` +
    `${model.identityDim} identity · ${model.expressionDim} expression`;

  const start = performance.now();
  const loop = () => {
    if (dirty) {
      dirty = false;
      const t0 = performance.now();
      viewer.update(state.identity, state.expression, state.rotations, state.translation, state.correctives);
      // Wall time only, and the GPU is asynchronous, so this measures the
      // cost of issuing the passes rather than of running them. It is the
      // number that matters for M3 regardless: it is what the main thread
      // loses per frame.
      const cost = performance.now() - t0;
      document.querySelector<HTMLElement>('#cost')!.textContent =
        `gpu dispatch ${cost.toFixed(2)} ms`;
    }
    viewer.render((performance.now() - start) / 1000);
    requestAnimationFrame(loop);
  };
  loop();
}

main().catch((error) => {
  document.querySelector<HTMLElement>('#status')!.textContent = String(error);
  throw error;
});
