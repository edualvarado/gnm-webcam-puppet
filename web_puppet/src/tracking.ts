/**
 * MediaPipe face tracking, wrapped so the rest of the app never sees the SDK.
 *
 * The landmarker is asked for all three outputs it can produce, because they
 * fail in different ways and the later modules need different ones:
 *
 *   landmarks    478 points, including the two 5-point iris rings. Geometric,
 *                so gaze can be read straight off them.
 *   blendshapes  52 ARKit-style scores from a trained classifier. Far more
 *                reliable than eyelid geometry, which is what defeated the
 *                earlier prototype's blink tracking entirely.
 *   transform    a 4x4 head pose. Getting this for free is what removes the
 *                per-frame pose solve the prototype ran.
 *
 * M3 only draws the first and derives gaze from it. The other two are carried
 * through now so M4 has them without touching this file again.
 */

import {
  FaceLandmarker,
  FilesetResolver,
  type FaceLandmarkerResult,
  type NormalizedLandmark,
} from '@mediapipe/tasks-vision';

/** Landmark indices of the two iris rings and the eye corners they sit in. */
export const EYE_LANDMARKS = {
  left: { irisCentre: 468, outerCorner: 33, innerCorner: 133 },
  right: { irisCentre: 473, outerCorner: 263, innerCorner: 362 },
} as const;

export interface Gaze {
  /** Iris centre in normalized image coordinates. */
  centre: NormalizedLandmark;
  /**
   * Iris offset from the eye's midpoint, as a fraction of eye half-width.
   * Roughly -1..1 horizontally; positive x is towards the image's right.
   */
  direction: { x: number; y: number };
}

export interface FaceFrame {
  landmarks: NormalizedLandmark[];
  blendshapes: Map<string, number>;
  /** Column-major 4x4 head pose, or null if the model did not produce one. */
  transform: Float32Array | null;
  gaze: { left: Gaze; right: Gaze } | null;
}

/**
 * Derives a gaze direction for one eye from iris and corner landmarks.
 *
 * Normalising by the eye's own width is what makes this survive the head
 * moving towards or away from the camera: both the offset and the width
 * scale together, so the ratio does not.
 */
function eyeGaze(
  landmarks: NormalizedLandmark[],
  indices: { irisCentre: number; outerCorner: number; innerCorner: number },
): Gaze {
  const iris = landmarks[indices.irisCentre];
  const outer = landmarks[indices.outerCorner];
  const inner = landmarks[indices.innerCorner];

  const midpointX = (outer.x + inner.x) / 2;
  const midpointY = (outer.y + inner.y) / 2;
  const halfWidth = Math.hypot(outer.x - inner.x, outer.y - inner.y) / 2;
  const scale = halfWidth > 1e-6 ? halfWidth : 1;

  return {
    centre: iris,
    direction: {
      x: (iris.x - midpointX) / scale,
      // The eye is far shorter than it is wide, so the same normaliser would
      // make vertical gaze read as enormous. Halving it keeps the two axes
      // comparable enough to draw as one vector.
      y: (iris.y - midpointY) / (scale * 0.5),
    },
  };
}

export class FaceTracker {
  private readonly landmarker: FaceLandmarker;
  private lastTimestamp = -1;

  private constructor(landmarker: FaceLandmarker) {
    this.landmarker = landmarker;
  }

  /**
   * Loads the WASM runtime and the model.
   *
   * @param wasmPath Directory holding the vendored MediaPipe WASM files.
   * @param modelPath Path to `face_landmarker.task`.
   */
  static async create(
    wasmPath = 'mediapipe/wasm',
    modelPath = 'assets/face_landmarker.task',
  ): Promise<FaceTracker> {
    const fileset = await FilesetResolver.forVisionTasks(wasmPath);
    const landmarker = await FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: modelPath, delegate: 'GPU' },
      runningMode: 'VIDEO',
      numFaces: 1,
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true,
    });
    return new FaceTracker(landmarker);
  }

  /**
   * Runs detection on the current video frame.
   *
   * @param video The playing video element.
   * @param timestampMs Monotonically increasing frame timestamp.
   * @returns The detected face, or null if none was found this frame.
   */
  detect(video: HTMLVideoElement, timestampMs: number): FaceFrame | null {
    // MediaPipe's video mode rejects a timestamp that does not advance, which
    // happens whenever the render loop outruns the camera's frame rate.
    if (timestampMs <= this.lastTimestamp) return null;
    this.lastTimestamp = timestampMs;

    const result: FaceLandmarkerResult = this.landmarker.detectForVideo(
      video, timestampMs,
    );
    const landmarks = result.faceLandmarks?.[0];
    if (!landmarks || landmarks.length === 0) return null;

    const blendshapes = new Map<string, number>();
    for (const category of result.faceBlendshapes?.[0]?.categories ?? []) {
      blendshapes.set(category.categoryName, category.score);
    }

    const matrix = result.facialTransformationMatrixes?.[0];

    return {
      landmarks,
      blendshapes,
      transform: matrix ? Float32Array.from(matrix.data) : null,
      gaze: {
        left: eyeGaze(landmarks, EYE_LANDMARKS.left),
        right: eyeGaze(landmarks, EYE_LANDMARKS.right),
      },
    };
  }

  close(): void {
    this.landmarker.close();
  }
}

/** Contour connector sets, exposed so the overlay does not import the SDK. */
export const CONTOURS = {
  faceOval: FaceLandmarker.FACE_LANDMARKS_FACE_OVAL,
  leftEye: FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
  rightEye: FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
  leftBrow: FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW,
  rightBrow: FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW,
  leftIris: FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS,
  rightIris: FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS,
  lips: FaceLandmarker.FACE_LANDMARKS_LIPS,
} as const;
