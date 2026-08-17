/**
 * Drawing tracked landmarks onto a 2D canvas.
 *
 * Shared by the full-frame tracker page and the circular picture-in-picture
 * in the viewer. The only thing that differs between them is how normalized
 * landmark coordinates land in canvas pixels, so that is the one thing this
 * takes as a parameter.
 */

import { CONTOURS, type FaceFrame } from './tracking.ts';

export const OVERLAY_COLORS = {
  points: 'rgba(124, 222, 197, 0.55)',
  contour: 'rgba(84, 196, 167, 0.35)',
  eye: '#54c4a7',
  iris: '#7cdec5',
  gaze: '#9b3ca1',
} as const;

/**
 * Maps normalized landmark coordinates to canvas pixels.
 *
 * A plain scale plus offset covers both cases: the tracker page draws the
 * whole frame, and the picture-in-picture draws a centre-cropped square, which
 * is the same mapping with a negative offset.
 */
export interface Projection {
  scaleX: number;
  scaleY: number;
  offsetX: number;
  offsetY: number;
}

export function fullFrame(width: number, height: number): Projection {
  return { scaleX: width, scaleY: height, offsetX: 0, offsetY: 0 };
}

/**
 * Projection for a centre-cropped square drawn at `size` pixels.
 *
 * @param videoWidth Source video width.
 * @param videoHeight Source video height.
 * @param size Edge length of the square destination.
 */
export function centreCrop(
  videoWidth: number,
  videoHeight: number,
  size: number,
): Projection & { sourceX: number; sourceY: number; side: number } {
  const side = Math.min(videoWidth, videoHeight);
  const sourceX = (videoWidth - side) / 2;
  const sourceY = (videoHeight - side) / 2;
  const scale = size / side;
  return {
    sourceX,
    sourceY,
    side,
    scaleX: videoWidth * scale,
    scaleY: videoHeight * scale,
    offsetX: -sourceX * scale,
    offsetY: -sourceY * scale,
  };
}

/**
 * The landmark-to-mesh mapping, drawn in the colours the mesh uses.
 *
 * When this is supplied the generic point cloud is replaced by just the
 * mapped landmarks, each in its assigned colour. That is the whole trick
 * behind reading the correspondence: the same point is the same colour in the
 * camera view and on the head, so matching them needs no labels.
 */
export interface CorrespondenceOverlay {
  /** MediaPipe landmark indices that have a mesh vertex. */
  landmarks: Uint16Array;
  /** RGB bytes per entry, parallel to `landmarks`. */
  colors: Uint8Array;
}

export interface OverlayOptions {
  /** Draw all 478 landmark points, not just the contours. */
  showPoints?: boolean;
  /** Length of the gaze ray in pixels, at a gaze magnitude of 1. */
  gazeScale?: number;
  /** Multiplier on every stroke width, for small canvases. */
  lineScale?: number;
  /** Draw the mapped landmarks in their mesh colours instead. */
  correspondence?: CorrespondenceOverlay | null;
}

function strokeContour(
  context: CanvasRenderingContext2D,
  frame: FaceFrame,
  connectors: readonly { start: number; end: number }[],
  projection: Projection,
): void {
  context.beginPath();
  for (const { start, end } of connectors) {
    const from = frame.landmarks[start];
    const to = frame.landmarks[end];
    if (!from || !to) continue;
    context.moveTo(
      from.x * projection.scaleX + projection.offsetX,
      from.y * projection.scaleY + projection.offsetY,
    );
    context.lineTo(
      to.x * projection.scaleX + projection.offsetX,
      to.y * projection.scaleY + projection.offsetY,
    );
  }
  context.stroke();
}

/** Draws the landmark overlay for one detected face. */
export function drawFace(
  context: CanvasRenderingContext2D,
  frame: FaceFrame,
  projection: Projection,
  options: OverlayOptions = {},
): void {
  const {
    showPoints = true, gazeScale = 70, lineScale = 1, correspondence = null,
  } = options;

  if (correspondence) {
    const radius = Math.max(1.2, 1.9 * lineScale);
    for (let i = 0; i < correspondence.landmarks.length; i++) {
      const point = frame.landmarks[correspondence.landmarks[i]];
      if (!point) continue;
      const r = correspondence.colors[i * 3];
      const g = correspondence.colors[i * 3 + 1];
      const b = correspondence.colors[i * 3 + 2];
      context.fillStyle = `rgb(${r},${g},${b})`;
      context.beginPath();
      context.arc(
        point.x * projection.scaleX + projection.offsetX,
        point.y * projection.scaleY + projection.offsetY,
        radius,
        0,
        Math.PI * 2,
      );
      context.fill();
    }
  } else if (showPoints) {
    // Small and dim: at 478 points the value is in watching the lattice hold
    // onto the face, not in reading any single dot.
    const dot = Math.max(0.8, 1.4 * lineScale);
    context.fillStyle = OVERLAY_COLORS.points;
    for (const point of frame.landmarks) {
      context.fillRect(
        point.x * projection.scaleX + projection.offsetX - dot / 2,
        point.y * projection.scaleY + projection.offsetY - dot / 2,
        dot,
        dot,
      );
    }
  }

  context.lineWidth = lineScale;
  context.strokeStyle = OVERLAY_COLORS.contour;
  for (const contour of [
    CONTOURS.faceOval, CONTOURS.lips, CONTOURS.leftBrow, CONTOURS.rightBrow,
  ]) {
    strokeContour(context, frame, contour, projection);
  }

  context.lineWidth = 1.5 * lineScale;
  context.strokeStyle = OVERLAY_COLORS.eye;
  for (const contour of [CONTOURS.leftEye, CONTOURS.rightEye]) {
    strokeContour(context, frame, contour, projection);
  }

  context.strokeStyle = OVERLAY_COLORS.iris;
  for (const contour of [CONTOURS.leftIris, CONTOURS.rightIris]) {
    strokeContour(context, frame, contour, projection);
  }

  if (!frame.gaze) return;
  for (const eye of [frame.gaze.left, frame.gaze.right]) {
    const x = eye.centre.x * projection.scaleX + projection.offsetX;
    const y = eye.centre.y * projection.scaleY + projection.offsetY;

    context.fillStyle = OVERLAY_COLORS.gaze;
    context.beginPath();
    context.arc(x, y, 2.5 * lineScale, 0, Math.PI * 2);
    context.fill();

    context.strokeStyle = OVERLAY_COLORS.gaze;
    context.lineWidth = 2 * lineScale;
    context.beginPath();
    context.moveTo(x, y);
    context.lineTo(
      x + eye.direction.x * gazeScale,
      y + eye.direction.y * gazeScale,
    );
    context.stroke();
  }
}
