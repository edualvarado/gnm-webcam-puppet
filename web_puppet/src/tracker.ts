/**
 * M3: the webcam with its landmarks drawn on it. Nothing else.
 *
 * Deliberately not wired to the GNM head. Tracking and retargeting fail in
 * different ways -- one as jittery or missing points, the other as a face that
 * moves wrongly -- and debugging them together means never knowing which one
 * is at fault. The head arrives in M4, once this is trusted.
 */

import { drawFace, fullFrame } from './overlay.ts';
import { FaceTracker, type FaceFrame } from './tracking.ts';

/** How far to draw the gaze ray, in pixels at a gaze magnitude of 1. */
const GAZE_RAY_SCALE = 70;

interface Elements {
  video: HTMLVideoElement;
  canvas: HTMLCanvasElement;
  status: HTMLElement;
  readout: HTMLElement;
  start: HTMLButtonElement;
  mirror: HTMLInputElement;
  showMesh: HTMLInputElement;
}

function formatReadout(frame: FaceFrame | null, fps: number): string {
  if (!frame) return `no face · ${fps.toFixed(0)} fps`;

  const gaze = frame.gaze!;
  const blink = (name: string) => (frame.blendshapes.get(name) ?? 0).toFixed(2);

  return [
    `${frame.landmarks.length} landmarks · ${fps.toFixed(0)} fps`,
    `gaze L  x ${gaze.left.direction.x.toFixed(2).padStart(5)}  ` +
      `y ${gaze.left.direction.y.toFixed(2).padStart(5)}`,
    `gaze R  x ${gaze.right.direction.x.toFixed(2).padStart(5)}  ` +
      `y ${gaze.right.direction.y.toFixed(2).padStart(5)}`,
    `blink   L ${blink('eyeBlinkLeft')}  R ${blink('eyeBlinkRight')}`,
  ].join('\n');
}

async function main(): Promise<void> {
  const el: Elements = {
    video: document.querySelector('#video')!,
    canvas: document.querySelector('#overlay')!,
    status: document.querySelector('#status')!,
    readout: document.querySelector('#readout')!,
    start: document.querySelector('#start')!,
    mirror: document.querySelector('#mirror')!,
    showMesh: document.querySelector('#showMesh')!,
  };

  const context = el.canvas.getContext('2d')!;

  el.status.textContent = 'Loading MediaPipe…';
  let tracker: FaceTracker;
  try {
    tracker = await FaceTracker.create();
  } catch (error) {
    el.status.textContent = `Failed to load MediaPipe: ${error}`;
    throw error;
  }
  el.status.textContent = 'Ready — click Start camera.';
  el.start.disabled = false;

  el.start.addEventListener('click', async () => {
    el.start.disabled = true;
    el.status.textContent = 'Requesting camera…';

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720 },
      });
    } catch (error) {
      el.status.textContent = `Camera unavailable: ${error}`;
      el.start.disabled = false;
      return;
    }

    el.video.srcObject = stream;
    await el.video.play();
    el.status.textContent = 'Tracking';

    let smoothedFps = 0;
    let previous = performance.now();

    const loop = () => {
      const now = performance.now();
      const delta = now - previous;
      previous = now;
      if (delta > 0) smoothedFps = smoothedFps * 0.9 + (1000 / delta) * 0.1;

      const width = el.video.videoWidth;
      const height = el.video.videoHeight;
      if (width && height) {
        if (el.canvas.width !== width || el.canvas.height !== height) {
          el.canvas.width = width;
          el.canvas.height = height;
        }

        const frame = tracker.detect(el.video, now);

        context.save();
        context.clearRect(0, 0, width, height);
        if (el.mirror.checked) {
          // Mirroring the canvas mirrors the overlay with the image, so the
          // landmarks stay on the face rather than needing their own flip.
          context.translate(width, 0);
          context.scale(-1, 1);
        }
        context.drawImage(el.video, 0, 0, width, height);
        if (frame) {
          drawFace(context, frame, fullFrame(width, height), {
            showPoints: el.showMesh.checked,
            gazeScale: GAZE_RAY_SCALE,
          });
        }
        context.restore();

        if (frame || smoothedFps > 0) {
          el.readout.textContent = formatReadout(frame, smoothedFps);
        }
      }
      requestAnimationFrame(loop);
    };
    loop();
  });
}

main().catch((error) => {
  document.querySelector<HTMLElement>('#status')!.textContent = String(error);
  throw error;
});
