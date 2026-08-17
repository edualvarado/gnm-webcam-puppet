/**
 * A floating circular webcam view with the tracking overlay on it.
 *
 * This exists so the head and the tracker can be watched at the same time
 * without being connected to each other yet. It owns the camera, the tracker
 * and its own draw loop, and exposes the latest frame -- so when M4 comes to
 * drive the head, it consumes `latest` and nothing here has to change.
 */

import { centreCrop, drawFace, type CorrespondenceOverlay } from './overlay.ts';
import { FaceTracker, type FaceFrame } from './tracking.ts';

export interface FacePipOptions {
  /** Diameter in CSS pixels. */
  size?: number;
  /** Mirror the view, as a selfie camera would. */
  mirror?: boolean;
  showPoints?: boolean;
}

export class FacePip {
  /** The most recent detection, or null if none yet. Read by later modules. */
  latest: FaceFrame | null = null;
  /** Smoothed detection rate, for display. */
  fps = 0;

  private readonly context: CanvasRenderingContext2D;
  private readonly video: HTMLVideoElement;
  private readonly size: number;
  private readonly showPoints: boolean;

  private tracker: FaceTracker | null = null;
  private stream: MediaStream | null = null;
  private running = false;
  private previous = 0;

  mirror: boolean;
  /** When set, mapped landmarks are drawn in their mesh colours. */
  correspondence: CorrespondenceOverlay | null = null;

  constructor(canvas: HTMLCanvasElement, options: FacePipOptions = {}) {
    const { size = 200, mirror = true, showPoints = true } = options;
    this.size = size;
    this.mirror = mirror;
    this.showPoints = showPoints;

    // Backing store at device resolution; the element stays `size` CSS px.
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = size * ratio;
    canvas.height = size * ratio;
    canvas.style.width = `${size}px`;
    canvas.style.height = `${size}px`;

    this.context = canvas.getContext('2d')!;
    this.context.scale(ratio, ratio);

    this.video = document.createElement('video');
    this.video.playsInline = true;
    this.video.muted = true;
  }

  get isRunning(): boolean {
    return this.running;
  }

  /**
   * The camera's aspect ratio, or 0 before the first frame arrives.
   *
   * The identity fit needs it because MediaPipe normalizes x by width and y by
   * height, so its landmarks are anisotropic on any non-square frame.
   */
  get aspect(): number {
    const { videoWidth, videoHeight } = this.video;
    return videoWidth && videoHeight ? videoWidth / videoHeight : 0;
  }

  /**
   * Loads the model if needed, opens the camera and starts drawing.
   *
   * @throws If the model fails to load or camera permission is refused.
   */
  async start(): Promise<void> {
    if (this.running) return;

    if (!this.tracker) this.tracker = await FaceTracker.create();

    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720 },
    });
    this.video.srcObject = this.stream;
    await this.video.play();

    this.running = true;
    this.previous = performance.now();
    this.loop();
  }

  /** Stops drawing and releases the camera. */
  stop(): void {
    this.running = false;
    for (const track of this.stream?.getTracks() ?? []) track.stop();
    this.stream = null;
    this.video.srcObject = null;
    this.latest = null;
    this.context.clearRect(0, 0, this.size, this.size);
  }

  private loop = (): void => {
    if (!this.running) return;

    const now = performance.now();
    const delta = now - this.previous;
    this.previous = now;
    if (delta > 0) this.fps = this.fps * 0.9 + (1000 / delta) * 0.1;

    const width = this.video.videoWidth;
    const height = this.video.videoHeight;
    if (width && height && this.tracker) {
      const frame = this.tracker.detect(this.video, now);
      // detect() returns null when the timestamp has not advanced, which is
      // not the same as "no face", so the previous frame is kept rather than
      // making the overlay flicker at every render that outruns the camera.
      if (frame) this.latest = frame;

      const crop = centreCrop(width, height, this.size);

      this.context.save();
      this.context.clearRect(0, 0, this.size, this.size);
      this.context.beginPath();
      this.context.arc(
        this.size / 2, this.size / 2, this.size / 2, 0, Math.PI * 2,
      );
      this.context.clip();

      if (this.mirror) {
        this.context.translate(this.size, 0);
        this.context.scale(-1, 1);
      }
      this.context.drawImage(
        this.video,
        crop.sourceX, crop.sourceY, crop.side, crop.side,
        0, 0, this.size, this.size,
      );
      if (this.latest) {
        drawFace(this.context, this.latest, crop, {
          showPoints: this.showPoints,
          correspondence: this.correspondence,
          // Scale the ray and the strokes with the circle so a 200 px view is
          // as legible as a 720 px one.
          gazeScale: this.size * 0.28,
          lineScale: this.size / 320,
        });
      }
      this.context.restore();
    }

    requestAnimationFrame(this.loop);
  };
}
