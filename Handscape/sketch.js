const INDEX_FINGER_TIP = 8;
const INDEX_FINGER_DIP = 7;
const INDEX_FINGER_PIP = 6;
const THUMB_TIP = 4;
const WRIST = 0;
// Fingers used for fist detection: [tip, pip] landmark pairs for middle,
// ring, and pinky (index is left out so a pinch doesn't read as a fist).
const FIST_FINGERS = [
  [12, 10],
  [16, 14],
  [20, 18],
];
// A finger counts as curled when its tip is closer to the wrist than its
// middle joint. Hysteresis: fist starts below the close ratio, ends above
// the open ratio.
const FIST_CLOSE_RATIO = 0.95;
const FIST_OPEN_RATIO = 1.05;
const MEDIAPIPE_VERSION = '0.10.14';
const NUM_HANDS = 2;
const PINCH_START_THRESHOLD = 0.05;
const PINCH_END_THRESHOLD = 0.08;
const HAND_LOST_GRACE_MS = 400;
// MediaPipe's default confidence (0.5): lower values keep cap-covered
// hands tracked longer but the low-confidence landmarks jitter, which
// makes drawn lines wobbly. Stability wins.
const DETECTION_CONFIDENCE = 0.5;
const MIN_POINT_DISTANCE = 4;
const RESAMPLE_SPACING = 10;
const SMOOTH_ITERATIONS = 2;
// Strokes drawn within this gap of each other join the same shape; after
// the gap passes with no new pinch, the shape is finalized as done.
const SHAPE_GROUP_GAP_MS = 3000;
// Moving shapes with the hand: touching a finished shape with the
// presence point grabs it; it lets go once the hand gets far away.
const TOUCH_RADIUS = 80;
const RELEASE_RADIUS = 180;
const DRAG_FOLLOW_STRENGTH = 0.15;
// Walk animation while a shape is being dragged. Strides are tied to
// distance traveled so steps match the ground; everything fades to fully
// static when the shape stops.
const WALK_PHASE_PER_PX = 0.045; // stride phase advance per px of travel
const WALK_SPEED_FOR_FULL = 220; // px/s of drag speed for full-strength walk
const WALK_BOB_AMOUNT = 6; // vertical bounce per step, in px
const WALK_TILT_AMOUNT = 0.09; // max side-to-side lean in radians
const WALK_LEG_SWING = 14; // max horizontal swing of the lowest points, in px
const WALK_AMOUNT_EASE = 6; // how fast the walk fades in/out, per second
// Shapes pushed past the canvas edge ease gently back inside instead of
// getting lost off-screen.
const EDGE_MARGIN = 24;
const EDGE_RETURN_RATE = 6; // per second
// Strokes glow like light: outer halo in the stroke color, brighter core.
const GLOW_BLUR = 18;
// Clear gesture: sweeping a closed fist horizontally across the screen
// wipes all shapes (the fist can't draw or grab, so it can't misfire).
const CLEAR_SWIPE_SPAN = 0.45; // fraction of screen width the fist must cover
const CLEAR_SWIPE_WINDOW_MS = 600; // within this much time
const CLEAR_SWIPE_FLATNESS = 2.5; // horizontal travel must beat vertical by this
const CURSOR_FOLLOW_RATE = 22;
const CURSOR_STYLE_RATE = 14;
const FILTER_MIN_CUTOFF = 1.5;
const FILTER_BETA = 0.01;
const FILTER_D_CUTOFF = 1.0;

// Finger-cap reference colors. `hue` (0-360) is what classification
// matches against — tune these to the caps under your lighting. `rgb` is
// the display color used for drawing (kept bright for the dark ground
// rather than using the raw camera sample).
const CAP_COLORS = [
  { name: 'green', hue: 120, rgb: [90, 230, 130] },
  { name: 'yellow', hue: 55, rgb: [255, 220, 80] },
  { name: 'red', hue: 358, rgb: [255, 90, 90] },
  { name: 'blue', hue: 215, rgb: [100, 170, 255] },
  { name: 'purple', hue: 280, rgb: [200, 120, 255] },
];
const DEFAULT_CAP = CAP_COLORS[1]; // before any classification: yellow
// White cap: no hue, so it's classified by brightness instead — very
// bright and nearly colorless. The value bar is set high so a gray wall
// doesn't pass; if a brightly lit white wall ever false-positives, raise
// WHITE_MIN_VALUE, and if the white cap won't register in dim light,
// lower it.
const WHITE_CAP = { name: 'white', rgb: [240, 244, 255] };
const WHITE_MIN_VALUE = 0.75;
const WHITE_MAX_SATURATION = 0.16;
// A sample too gray or too dark has no meaningful hue — keep the previous
// classification instead of guessing from noise.
const CAP_MIN_SATURATION = 0.18;
const CAP_MIN_VALUE = 0.1;
// Skin sits in this hue band (orange-ish) — reject it outright so a bare
// hand can never register as a cap. This is why the cap set has red
// instead of orange.
const SKIN_HUE_MIN = 10;
const SKIN_HUE_MAX = 42;
// Red lives just below the skin band, so it must be vividly saturated
// (felt-level, not skin-level) to count.
const RED_MIN_SATURATION = 0.5;
// A sample must land this close in hue to some reference to classify at
// all; anything further is noise, not a cap.
const MAX_HUE_MATCH_DISTANCE = 40;
// A new cap color is only accepted after this many consecutive agreeing
// detection frames, so one bad sample can't flip the color mid-session.
const CAP_CONSISTENT_FRAMES = 4;
// Averaging window: center +/- this many pixels (9x9 total — the felt
// caps are large, so a wide window is safe and much more stable).
const COLOR_SAMPLE_RADIUS = 4;
// The caps cover the fingertip, so the tip landmark itself is the primary
// sample spot; a second candidate slightly past the tip (tip + (tip - dip)
// * this factor) covers caps that sit a little long on the finger.
const CAP_SAMPLE_EXTRAPOLATION = 0.25;
// Hand identity: A/B assignment sticks to recent positions (continuity)
// and falls back to MediaPipe's left/right handedness for new hands.
const IDENTITY_STICKY_MS = 1000;

let capture;
let handLandmarker = null;
let handTrackingReady = false;
let handTrackingError = null;
let lastVideoTime = -1;
let clearFlash = 0;
let showHandDebug = false;

// Offscreen copy of the current video frame for pixel sampling.
let sampleCanvas = null;
let sampleCtx = null;

// All drawn shapes, in draw order. Each shape is:
//   { strokes: [{ points: [{x, y}, ...], color: [r, g, b] }],
//     lastStrokeEndTime, done, offset }
// Stroke points are absolute canvas coordinates recorded at draw time and
// are never recalculated afterward; each stroke keeps the cap color of
// the hand that drew it.
let shapes = [];

class LowPassFilter {
  constructor() {
    this.initialized = false;
    this.value = 0;
  }

  filter(value, alpha) {
    if (!this.initialized) {
      this.initialized = true;
      this.value = value;
    } else {
      this.value = alpha * value + (1 - alpha) * this.value;
    }
    return this.value;
  }
}

// One Euro filter: heavy smoothing at low speeds (kills jitter), light
// smoothing at high speeds (keeps fast moves responsive).
class OneEuroFilter {
  constructor(minCutoff, beta, dCutoff) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.valueFilter = new LowPassFilter();
    this.derivativeFilter = new LowPassFilter();
    this.lastTime = null;
  }

  alpha(cutoff, dt) {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }

  filter(value, timeMs) {
    if (this.lastTime === null) {
      this.lastTime = timeMs;
    }
    const dt = Math.max((timeMs - this.lastTime) / 1000, 0.001);
    this.lastTime = timeMs;

    const previous = this.valueFilter.initialized ? this.valueFilter.value : value;
    const derivative = (value - previous) / dt;
    const smoothedDerivative = this.derivativeFilter.filter(
      derivative,
      this.alpha(this.dCutoff, dt)
    );
    const cutoff = this.minCutoff + this.beta * Math.abs(smoothedDerivative);
    return this.valueFilter.filter(value, this.alpha(cutoff, dt));
  }

  reset() {
    this.valueFilter.initialized = false;
    this.derivativeFilter.initialized = false;
    this.lastTime = null;
  }
}

// One independent state bundle per identity. Identities are assigned by
// position continuity (with handedness as the fallback for new hands),
// so a hand keeps its state (stroke in progress, held shape, filters)
// even when MediaPipe's internal ordering of the two hands flips.
function createHand(id, markerColor) {
  return {
    id,
    markerColor,
    lastSeenTime: -Infinity,
    // Where this hand was last detected. Unlike presencePoint this is
    // never nulled on hand loss — identity matching needs it for the
    // whole IDENTITY_STICKY_MS window, which outlives the grace period.
    lastKnownPosition: null,
    targetTip: null,
    presencePoint: null,
    isPinching: false,
    pinchDistanceSmoothed: null,
    isFist: false,
    currentStroke: null,
    currentShape: null,
    heldShape: null,
    grabDx: 0,
    grabDy: 0,
    swipeTrail: [],
    capColor: null, // classified CAP_COLORS entry
    pendingCap: null, // candidate cap awaiting consistent frames
    handednessLabel: null, // 'Right' / 'Left' from MediaPipe, for debug
    sampledIndexRgb: null, // raw averaged camera color, for the debug panel
    tipFilterX: new OneEuroFilter(FILTER_MIN_CUTOFF, FILTER_BETA, FILTER_D_CUTOFF),
    tipFilterY: new OneEuroFilter(FILTER_MIN_CUTOFF, FILTER_BETA, FILTER_D_CUTOFF),
    presenceFilterX: new OneEuroFilter(FILTER_MIN_CUTOFF, FILTER_BETA, FILTER_D_CUTOFF),
    presenceFilterY: new OneEuroFilter(FILTER_MIN_CUTOFF, FILTER_BETA, FILTER_D_CUTOFF),
    cursor: { x: 0, y: 0, alpha: 0, pinchAmount: 0, initialized: false },
  };
}

// Marker colors are deliberately not cap colors: A is a cool white-blue,
// B a warm rose, so identity never gets confused with drawing color.
const handA = createHand('A', [170, 210, 255]);
const handB = createHand('B', [255, 130, 160]);
const allHands = [handA, handB];

function setup() {
  createCanvas(windowWidth, windowHeight);
  capture = createCapture(VIDEO);
  capture.size(windowWidth, windowHeight);
  capture.hide();
  initHandTracking();
}

async function initHandTracking() {
  try {
    const { HandLandmarker, FilesetResolver } = await import(
      `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/vision_bundle.mjs`
    );

    const vision = await FilesetResolver.forVisionTasks(
      `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`
    );

    try {
      handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
          delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        numHands: NUM_HANDS,
        minHandDetectionConfidence: DETECTION_CONFIDENCE,
        minHandPresenceConfidence: DETECTION_CONFIDENCE,
        minTrackingConfidence: DETECTION_CONFIDENCE,
      });
    } catch (gpuError) {
      console.warn('GPU hand tracking failed, falling back to CPU:', gpuError);
      handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
          delegate: 'CPU',
        },
        runningMode: 'VIDEO',
        numHands: NUM_HANDS,
        minHandDetectionConfidence: DETECTION_CONFIDENCE,
        minHandPresenceConfidence: DETECTION_CONFIDENCE,
        minTrackingConfidence: DETECTION_CONFIDENCE,
      });
    }

    handTrackingReady = true;
    console.log('Hand tracking ready');
  } catch (error) {
    handTrackingError = error;
    console.error('Hand tracking failed to load:', error);
  }
}

function draw() {
  // The camera keeps running for hand tracking but is never drawn —
  // only the drawings and markers show, on a deep night-sky ground.
  background(7, 10, 28);

  updateHandTracking();
  for (const hand of allHands) {
    updateCursor(hand);
    updateShapeLifecycle(hand);
    updateShapeDragging(hand);
    updateClearGesture(hand);
  }
  keepShapesOnScreen();
  updateWalkCycles();
  drawShapes();
  drawClearFlash();
  for (const hand of allHands) {
    drawPresencePoint(hand);
    drawCursor(hand);
  }
  drawHandDebug();
  drawStatus();
}

// ---------------------------------------------------------------------
// Color sampling and classification
// ---------------------------------------------------------------------

// Copies the current (unmirrored) video frame into an offscreen canvas
// once per detection frame, so fingertip pixels can be read cheaply.
function refreshSampleFrame(video) {
  const w = video.videoWidth;
  const h = video.videoHeight;
  if (!w || !h) {
    return false;
  }

  if (!sampleCanvas || sampleCanvas.width !== w || sampleCanvas.height !== h) {
    sampleCanvas = document.createElement('canvas');
    sampleCanvas.width = w;
    sampleCanvas.height = h;
    sampleCtx = sampleCanvas.getContext('2d', { willReadFrequently: true });
  }

  sampleCtx.drawImage(video, 0, 0, w, h);
  return true;
}

// Averages a (2*COLOR_SAMPLE_RADIUS+1)^2 pixel block around a normalized
// landmark coordinate in the raw video frame. Returns [r, g, b] or null.
function sampleVideoColor(nx, ny) {
  if (!sampleCtx) {
    return null;
  }

  const w = sampleCanvas.width;
  const h = sampleCanvas.height;
  const cx = constrain(round(nx * w), COLOR_SAMPLE_RADIUS, w - 1 - COLOR_SAMPLE_RADIUS);
  const cy = constrain(round(ny * h), COLOR_SAMPLE_RADIUS, h - 1 - COLOR_SAMPLE_RADIUS);
  const size = COLOR_SAMPLE_RADIUS * 2 + 1;

  const pixels = sampleCtx.getImageData(
    cx - COLOR_SAMPLE_RADIUS,
    cy - COLOR_SAMPLE_RADIUS,
    size,
    size
  ).data;

  let r = 0;
  let g = 0;
  let b = 0;
  const count = size * size;
  for (let i = 0; i < pixels.length; i += 4) {
    r += pixels[i];
    g += pixels[i + 1];
    b += pixels[i + 2];
  }
  return [r / count, g / count, b / count];
}

// Standard RGB -> HSV. h in [0, 360), s and v in [0, 1].
function rgbToHsv(r, g, b) {
  const rn = r / 255;
  const gn = g / 255;
  const bn = b / 255;
  const maxC = Math.max(rn, gn, bn);
  const minC = Math.min(rn, gn, bn);
  const delta = maxC - minC;

  let h = 0;
  if (delta > 0) {
    if (maxC === rn) {
      h = 60 * (((gn - bn) / delta) % 6);
    } else if (maxC === gn) {
      h = 60 * ((bn - rn) / delta + 2);
    } else {
      h = 60 * ((rn - gn) / delta + 4);
    }
    if (h < 0) {
      h += 360;
    }
  }

  return { h, s: maxC === 0 ? 0 : delta / maxC, v: maxC };
}

function hueDistance(a, b) {
  const d = abs(a - b) % 360;
  return d > 180 ? 360 - d : d;
}

// Classifies a sampled fingertip color against the cap references by
// closest hue. Returns null (no cap) for anything gray, dark, in the
// skin-tone hue band, too far from every reference, or red-hued without
// felt-level saturation.
function classifyCapColor(rgb) {
  const { h, s, v } = rgbToHsv(rgb[0], rgb[1], rgb[2]);

  // White is the special case: bright and colorless. Checked before the
  // gray rejection below, which would otherwise discard it.
  if (v >= WHITE_MIN_VALUE && s <= WHITE_MAX_SATURATION) {
    return WHITE_CAP;
  }

  if (s < CAP_MIN_SATURATION || v < CAP_MIN_VALUE) {
    return null;
  }
  if (h > SKIN_HUE_MIN && h < SKIN_HUE_MAX) {
    return null;
  }

  let best = null;
  let bestDistance = Infinity;
  for (const cap of CAP_COLORS) {
    const d = hueDistance(h, cap.hue);
    if (d < bestDistance) {
      bestDistance = d;
      best = cap;
    }
  }

  if (bestDistance > MAX_HUE_MATCH_DISTANCE) {
    return null;
  }
  if (best.name === 'red' && s < RED_MIN_SATURATION) {
    return null;
  }
  return best;
}

// Samples the hand's index cap. The cone caps extend past the fingertip,
// so beyond the tip landmark itself we also try a point extrapolated
// outward along the finger's direction (dip joint -> tip -> onward into
// the cap), and keep whichever candidate is most saturated — i.e. most
// likely to be felt rather than skin or wall.
function sampleCapColor(hand, landmarks) {
  // Only sample from a clearly extended, non-pinching index finger: a
  // curled or pinching finger puts the tip landmark over skin, and the
  // stroke color is locked at pinch start anyway.
  if (hand.isPinching) {
    return;
  }
  const wrist = landmarks[WRIST];
  const tip = landmarks[INDEX_FINGER_TIP];
  const pip = landmarks[INDEX_FINGER_PIP];
  const tipReach = dist(tip.x, tip.y, wrist.x, wrist.y);
  const pipReach = max(dist(pip.x, pip.y, wrist.x, wrist.y), 0.001);
  if (tipReach < pipReach * 1.1) {
    hand.pendingCap = null;
    return;
  }

  const dip = landmarks[INDEX_FINGER_DIP];
  const beyond = {
    x: tip.x + (tip.x - dip.x) * CAP_SAMPLE_EXTRAPOLATION,
    y: tip.y + (tip.y - dip.y) * CAP_SAMPLE_EXTRAPOLATION,
  };

  // Tip first (the cap covers the fingertip), then the slightly extended
  // point. The first candidate that classifies as a cap wins — a vivid
  // background color behind the hand can't outvote the cap itself.
  let cap = null;
  let sampled = null;
  for (const spot of [tip, beyond]) {
    const rgb = sampleVideoColor(spot.x, spot.y);
    if (!rgb) {
      continue;
    }
    if (!sampled) {
      sampled = rgb;
    }
    const candidate = classifyCapColor(rgb);
    if (candidate) {
      cap = candidate;
      sampled = rgb;
      break;
    }
  }

  if (!sampled) {
    return;
  }

  hand.sampledIndexRgb = sampled;

  if (!cap || (hand.capColor && cap.name === hand.capColor.name)) {
    hand.pendingCap = null;
    return;
  }

  // A different cap has to persist for several consecutive detection
  // frames before it takes over — one stray sample changes nothing.
  if (hand.pendingCap && hand.pendingCap.cap.name === cap.name) {
    hand.pendingCap.count++;
    if (hand.pendingCap.count >= CAP_CONSISTENT_FRAMES) {
      hand.capColor = cap;
      hand.pendingCap = null;
    }
  } else {
    hand.pendingCap = { cap, count: 1 };
  }
}

// ---------------------------------------------------------------------
// Hand tracking and identity
// ---------------------------------------------------------------------

function updateHandTracking() {
  if (!handLandmarker || !capture || !capture.loadedmetadata) {
    return;
  }

  try {
    const video = capture.elt;
    if (video.readyState < 2 || video.currentTime === lastVideoTime) {
      return;
    }

    lastVideoTime = video.currentTime;
    const now = performance.now();
    const results = handLandmarker.detectForVideo(video, now);

    if (!results.landmarks || results.landmarks.length === 0) {
      return;
    }

    refreshSampleFrame(video);

    // Package detections with their screen-space centroid (for position
    // continuity) and MediaPipe's handedness label (for new hands).
    const detections = results.landmarks.slice(0, NUM_HANDS).map((landmarks, i) => {
      let sumX = 0;
      let sumY = 0;
      for (const landmark of landmarks) {
        sumX += landmark.x;
        sumY += landmark.y;
      }
      const category = results.handednesses?.[i]?.[0];
      return {
        landmarks,
        screenX: (1 - sumX / landmarks.length) * width,
        screenY: (sumY / landmarks.length) * height,
        label: category ? category.categoryName : null,
      };
    });

    if (detections.length >= 2) {
      const [d0, d1] = detections;
      const [h0, h1] = assignTwoHands(d0, d1);
      processHandDetection(h0, d0, now);
      processHandDetection(h1, d1, now);
    } else {
      const detection = detections[0];
      processHandDetection(assignSingleHand(detection), detection, now);
    }
  } catch (error) {
    console.error('Hand tracking frame error:', error);
  }
}

function distanceToHand(hand, detection) {
  if (!hand.lastKnownPosition) {
    return Infinity;
  }
  return dist(
    hand.lastKnownPosition.x,
    hand.lastKnownPosition.y,
    detection.screenX,
    detection.screenY
  );
}

function recentlySeen(hand) {
  return millis() - hand.lastSeenTime < IDENTITY_STICKY_MS;
}

// Identity for two simultaneous detections. Position continuity comes
// first — each detection goes to whichever hand it's nearest to, which
// keeps identities stable even when MediaPipe's array order flips or the
// hands cross. With only one hand known, the detection nearer its last
// position keeps that identity and the other detection becomes the new
// hand. Brand-new hands fall back to handedness (right hand = A), then
// to screen position.
function assignTwoHands(d0, d1) {
  const recentA = recentlySeen(handA);
  const recentB = recentlySeen(handB);

  if (recentA && recentB) {
    const straight = distanceToHand(handA, d0) + distanceToHand(handB, d1);
    const swapped = distanceToHand(handA, d1) + distanceToHand(handB, d0);
    return straight <= swapped ? [handA, handB] : [handB, handA];
  }

  if (recentA !== recentB) {
    const known = recentA ? handA : handB;
    const fresh = known === handA ? handB : handA;
    return distanceToHand(known, d0) <= distanceToHand(known, d1)
      ? [known, fresh]
      : [fresh, known];
  }

  if (d0.label && d1.label && d0.label !== d1.label) {
    return d0.label === 'Right' ? [handA, handB] : [handB, handA];
  }

  // Same/unknown labels (e.g. two people's right hands): leftmost on
  // screen is A.
  return d0.screenX <= d1.screenX ? [handA, handB] : [handB, handA];
}

// Identity for a lone detection: continuity with a recently seen hand
// first, then handedness for a fresh hand.
function assignSingleHand(detection) {
  const recentA = recentlySeen(handA);
  const recentB = recentlySeen(handB);

  if (recentA && !recentB) {
    return handA;
  }
  if (recentB && !recentA) {
    return handB;
  }
  if (recentA && recentB) {
    return distanceToHand(handA, detection) <= distanceToHand(handB, detection)
      ? handA
      : handB;
  }

  return detection.label === 'Left' ? handB : handA;
}

// Everything derived from one hand's landmarks for one frame: pinch,
// fist, filtered tip and presence, and cap-color sampling.
function processHandDetection(hand, detection, now) {
  const landmarks = detection.landmarks;
  hand.lastSeenTime = millis();
  hand.lastKnownPosition = { x: detection.screenX, y: detection.screenY };
  hand.handednessLabel = detection.label;

  const indexLandmark = landmarks[INDEX_FINGER_TIP];
  const thumbLandmark = landmarks[THUMB_TIP];

  sampleCapColor(hand, landmarks);

  const pinchDistance = dist(
    indexLandmark.x,
    indexLandmark.y,
    thumbLandmark.x,
    thumbLandmark.y
  );
  if (hand.pinchDistanceSmoothed === null) {
    hand.pinchDistanceSmoothed = pinchDistance;
  } else {
    hand.pinchDistanceSmoothed = lerp(hand.pinchDistanceSmoothed, pinchDistance, 0.5);
  }

  updateFistState(hand, landmarks);

  // Hysteresis: pinch must close below the start threshold to begin a
  // stroke and open past the end threshold to finish it, so a wobbly
  // pinch doesn't chop one stroke into many. A fist never counts as a
  // pinch, even though it brings the thumb and index tip together.
  const pinching =
    !hand.isFist &&
    (hand.isPinching
      ? hand.pinchDistanceSmoothed < PINCH_END_THRESHOLD
      : hand.pinchDistanceSmoothed < PINCH_START_THRESHOLD);

  const rawX = (1 - indexLandmark.x) * width;
  const rawY = indexLandmark.y * height;
  hand.targetTip = {
    x: hand.tipFilterX.filter(rawX, now),
    y: hand.tipFilterY.filter(rawY, now),
  };

  // Presence point: centroid of all 21 landmarks — the hand's overall
  // position, distinct from the fingertip used for drawing.
  let sumX = 0;
  let sumY = 0;
  for (const landmark of landmarks) {
    sumX += landmark.x;
    sumY += landmark.y;
  }
  const presenceRawX = (1 - sumX / landmarks.length) * width;
  const presenceRawY = (sumY / landmarks.length) * height;
  hand.presencePoint = {
    x: hand.presenceFilterX.filter(presenceRawX, now),
    y: hand.presenceFilterY.filter(presenceRawY, now),
  };

  if (pinching) {
    if (!hand.isPinching) {
      hand.isPinching = true;
      beginStroke(hand);
    }
  } else if (hand.isPinching) {
    finishCurrentStroke(hand);
  }
}

// Fist detection with hysteresis: closing middle, ring, and pinky toward
// the wrist enters the fist state; the hand must open clearly to leave it.
function updateFistState(hand, landmarks) {
  const wrist = landmarks[WRIST];
  let maxRatio = 0;
  let minRatio = Infinity;

  for (const [tipIndex, pipIndex] of FIST_FINGERS) {
    const tip = landmarks[tipIndex];
    const pip = landmarks[pipIndex];
    const tipDistance = dist(tip.x, tip.y, wrist.x, wrist.y);
    const pipDistance = max(dist(pip.x, pip.y, wrist.x, wrist.y), 0.001);
    const ratio = tipDistance / pipDistance;
    maxRatio = max(maxRatio, ratio);
    minRatio = min(minRatio, ratio);
  }

  if (!hand.isFist && maxRatio < FIST_CLOSE_RATIO) {
    hand.isFist = true;
  } else if (hand.isFist && minRatio > FIST_OPEN_RATIO) {
    hand.isFist = false;
  }
}

// ---------------------------------------------------------------------
// Per-hand cursor, strokes, and shapes
// ---------------------------------------------------------------------

// Runs every draw frame (not just on new camera frames), so the cursor
// glides between detections and strokes get points at full frame rate.
function updateCursor(hand) {
  const handLost = millis() - hand.lastSeenTime > HAND_LOST_GRACE_MS;

  if (handLost) {
    hand.targetTip = null;
    hand.presencePoint = null;
    finishCurrentStroke(hand);
    hand.tipFilterX.reset();
    hand.tipFilterY.reset();
    hand.presenceFilterX.reset();
    hand.presenceFilterY.reset();
    hand.pinchDistanceSmoothed = null;
    hand.isFist = false;
    hand.heldShape = null;
  }

  const dt = deltaTime / 1000;
  const styleEase = 1 - Math.exp(-dt * CURSOR_STYLE_RATE);
  const cursor = hand.cursor;
  cursor.alpha = lerp(cursor.alpha, hand.targetTip ? 1 : 0, styleEase);
  cursor.pinchAmount = lerp(cursor.pinchAmount, hand.isPinching ? 1 : 0, styleEase);

  if (!hand.targetTip) {
    return;
  }

  if (!cursor.initialized) {
    cursor.x = hand.targetTip.x;
    cursor.y = hand.targetTip.y;
    cursor.initialized = true;
  } else {
    const followEase = 1 - Math.exp(-dt * CURSOR_FOLLOW_RATE);
    cursor.x = lerp(cursor.x, hand.targetTip.x, followEase);
    cursor.y = lerp(cursor.y, hand.targetTip.y, followEase);
  }

  if (hand.isPinching && hand.currentStroke) {
    addPointToStroke(hand.currentStroke, cursor.x, cursor.y);
  }
}

function beginStroke(hand) {
  if (!hand.currentShape) {
    hand.currentShape = {
      strokes: [],
      lastStrokeEndTime: millis(),
      done: false,
      offset: { x: 0, y: 0 },
    };
    shapes.push(hand.currentShape);
  }

  // The stroke keeps the cap color its hand shows at pinch time.
  const cap = hand.capColor || DEFAULT_CAP;
  hand.currentStroke = { points: [], color: [...cap.rgb] };
  hand.currentShape.strokes.push(hand.currentStroke);
}

function addPointToStroke(stroke, x, y) {
  const lastPoint = stroke.points[stroke.points.length - 1];
  if (lastPoint && dist(lastPoint.x, lastPoint.y, x, y) < MIN_POINT_DISTANCE) {
    return;
  }

  stroke.points.push({ x, y });
}

// Ends the in-progress stroke. Its points get one final smoothing pass
// here and are then permanent — nothing recalculates them afterward.
function finishCurrentStroke(hand) {
  hand.isPinching = false;

  if (!hand.currentStroke) {
    return;
  }

  if (hand.currentStroke.points.length === 0 && hand.currentShape) {
    const strokeList = hand.currentShape.strokes;
    strokeList.splice(strokeList.indexOf(hand.currentStroke), 1);
  } else {
    hand.currentStroke.points = smoothStroke(hand.currentStroke.points);
  }

  if (hand.currentShape) {
    hand.currentShape.lastStrokeEndTime = millis();
  }

  hand.currentStroke = null;
}

// Finalizes a hand's current shape once SHAPE_GROUP_GAP_MS passes without
// a new pinch; that hand's next stroke then starts a fresh shape.
function updateShapeLifecycle(hand) {
  if (!hand.currentShape || hand.isPinching) {
    return;
  }

  if (millis() - hand.currentShape.lastStrokeEndTime > SHAPE_GROUP_GAP_MS) {
    if (hand.currentShape.strokes.length === 0) {
      shapes.splice(shapes.indexOf(hand.currentShape), 1);
    } else {
      finalizeShape(hand.currentShape);
    }
    hand.currentShape = null;
  }
}

// Marks a shape done and caches its geometry and walk-animation state.
function finalizeShape(shape) {
  let sumX = 0;
  let sumY = 0;
  let count = 0;
  const bounds = { minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity };
  for (const shapeStroke of shape.strokes) {
    for (const point of shapeStroke.points) {
      sumX += point.x;
      sumY += point.y;
      count++;
      bounds.minX = min(bounds.minX, point.x);
      bounds.minY = min(bounds.minY, point.y);
      bounds.maxX = max(bounds.maxX, point.x);
      bounds.maxY = max(bounds.maxY, point.y);
    }
  }

  shape.centroid = { x: sumX / count, y: sumY / count };
  shape.bounds = bounds;
  shape.walkPhase = 0;
  shape.walkAmount = 0;
  shape.speed = 0;
  shape.prevOffset = { ...shape.offset };
  shape.done = true;
}

// Advances each finished shape's walk cycle from how far it actually
// moved this frame (i.e. how the hand is dragging it): the stride phase
// tracks distance traveled, and the animation strength eases up with drag
// speed and back down to zero at rest.
function updateWalkCycles() {
  const dt = deltaTime / 1000;
  if (dt <= 0) {
    return;
  }

  for (const shape of shapes) {
    if (!shape.done) {
      continue;
    }

    const moved = dist(
      shape.offset.x,
      shape.offset.y,
      shape.prevOffset.x,
      shape.prevOffset.y
    );
    shape.prevOffset.x = shape.offset.x;
    shape.prevOffset.y = shape.offset.y;

    shape.speed = lerp(shape.speed, moved / dt, 0.2);
    shape.walkPhase += moved * WALK_PHASE_PER_PX;

    const targetAmount = constrain(shape.speed / WALK_SPEED_FOR_FULL, 0, 1);
    const ease = 1 - Math.exp(-dt * WALK_AMOUNT_EASE);
    shape.walkAmount = lerp(shape.walkAmount, targetAmount, ease);
  }
}

function clearAllShapes() {
  shapes = [];
  for (const hand of allHands) {
    hand.currentShape = null;
    hand.currentStroke = null;
    hand.heldShape = null;
    hand.isPinching = false;
    hand.swipeTrail = [];
  }
}

// Detects the clear gesture: a closed fist sweeping mostly horizontally
// across CLEAR_SWIPE_SPAN of the screen within CLEAR_SWIPE_WINDOW_MS.
// The trail only accumulates while the fist is closed, so opening the
// hand (or losing it) resets the gesture. Either hand can clear.
function updateClearGesture(hand) {
  if (!hand.isFist || !hand.presencePoint) {
    hand.swipeTrail = [];
    return;
  }

  const now = millis();
  hand.swipeTrail.push({ x: hand.presencePoint.x, y: hand.presencePoint.y, t: now });
  while (hand.swipeTrail.length > 0 && now - hand.swipeTrail[0].t > CLEAR_SWIPE_WINDOW_MS) {
    hand.swipeTrail.shift();
  }

  const first = hand.swipeTrail[0];
  const last = hand.swipeTrail[hand.swipeTrail.length - 1];
  const dx = last.x - first.x;
  const dy = last.y - first.y;

  if (abs(dx) >= width * CLEAR_SWIPE_SPAN && abs(dx) > abs(dy) * CLEAR_SWIPE_FLATNESS) {
    clearAllShapes();
    clearFlash = 1;
  }
}

// Brief full-screen flash so a gesture-triggered clear reads as
// deliberate feedback rather than the drawings just vanishing.
function drawClearFlash() {
  if (clearFlash < 0.01) {
    return;
  }

  noStroke();
  fill(200, 220, 255, 90 * clearFlash);
  rect(0, 0, width, height);
  clearFlash *= Math.exp(-(deltaTime / 1000) * 5);
}

// Eases any shape that ends up past the canvas edge gently back inside,
// so nothing can be pushed off-screen and lost. While a shape is being
// dragged outward this reads as a soft bounce/resistance at the border.
function keepShapesOnScreen() {
  const dt = deltaTime / 1000;
  if (dt <= 0) {
    return;
  }
  const ease = 1 - Math.exp(-dt * EDGE_RETURN_RATE);

  for (const shape of shapes) {
    if (!shape.done) {
      continue;
    }

    const { minX, minY, maxX, maxY } = shape.bounds;
    let lowX = EDGE_MARGIN - minX;
    let highX = width - EDGE_MARGIN - maxX;
    let lowY = EDGE_MARGIN - minY;
    let highY = height - EDGE_MARGIN - maxY;

    // A shape larger than the canvas can't fit inside the margins; aim
    // for centered instead of fighting between the two limits.
    if (lowX > highX) {
      lowX = highX = (lowX + highX) / 2;
    }
    if (lowY > highY) {
      lowY = highY = (lowY + highY) / 2;
    }

    const clampedX = constrain(shape.offset.x, lowX, highX);
    const clampedY = constrain(shape.offset.y, lowY, highY);
    shape.offset.x = lerp(shape.offset.x, clampedX, ease);
    shape.offset.y = lerp(shape.offset.y, clampedY, ease);
  }
}

// Walk transform for one point, in the shape's local (pre-offset) space:
// a lean that rotates around the centroid, a bob with each step, and a
// leg swing that shears points below the "hips" (the centroid) back and
// forth — strongest at the lowest points, zero at the hips.
function walkTransformPoint(shape, point) {
  const amount = shape.walkAmount;
  const tilt = sin(shape.walkPhase) * WALK_TILT_AMOUNT * amount;
  const bobY = -abs(sin(shape.walkPhase)) * WALK_BOB_AMOUNT * amount;
  const cosTilt = cos(tilt);
  const sinTilt = sin(tilt);
  const centerX = shape.centroid.x;
  const centerY = shape.centroid.y;

  const legDepth = max(shape.bounds.maxY - centerY, 1);
  const depth = constrain((point.y - centerY) / legDepth, 0, 1);
  const swing = sin(shape.walkPhase) * WALK_LEG_SWING * amount * depth;

  const relX = point.x - centerX;
  const relY = point.y - centerY;
  return {
    x: centerX + relX * cosTilt - relY * sinTilt + swing,
    y: centerY + relX * sinTilt + relY * cosTilt + bobY,
  };
}

// Shortest distance from a point to any recorded point of the shape,
// taking the shape's current offset into account.
function distanceToShape(shape, x, y) {
  let closest = Infinity;
  for (const shapeStroke of shape.strokes) {
    for (const point of shapeStroke.points) {
      const d = dist(point.x + shape.offset.x, point.y + shape.offset.y, x, y);
      if (d < closest) {
        closest = d;
      }
    }
  }
  return closest;
}

// The hand counts as "on" a shape when it's inside the shape's bounding
// box (so hovering in the middle of a drawn circle works, not just on
// its outline) or within `radius` of the outline. Grab score of 0 means
// directly on/inside; otherwise it's the outline distance.
function grabScore(shape, x, y, radius) {
  const localX = x - shape.offset.x;
  const localY = y - shape.offset.y;
  const { minX, minY, maxX, maxY } = shape.bounds;

  if (localX >= minX && localX <= maxX && localY >= minY && localY <= maxY) {
    return 0;
  }

  const outlineDistance = distanceToShape(shape, x, y);
  return outlineDistance < radius ? outlineDistance : Infinity;
}

// Touching a finished shape with the presence point grabs it; while held
// it eases after the hand (keeping the spot you grabbed it by), and it is
// released when the hand moves well away or leaves the frame. Movement is
// stored as an explicit per-shape offset — the recorded points themselves
// are never modified. Each hand can hold its own shape, but never one the
// other hand is already holding.
function updateShapeDragging(hand) {
  if (hand.isFist) {
    hand.heldShape = null;
    return;
  }

  if (!hand.presencePoint || hand.isPinching) {
    if (!hand.presencePoint) {
      hand.heldShape = null;
    }
    return;
  }

  if (
    hand.heldShape &&
    grabScore(hand.heldShape, hand.presencePoint.x, hand.presencePoint.y, RELEASE_RADIUS) ===
      Infinity
  ) {
    hand.heldShape = null;
  }

  if (!hand.heldShape) {
    const otherHand = hand === handA ? handB : handA;
    let closestShape = null;
    let closestScore = Infinity;
    for (const shape of shapes) {
      if (!shape.done || shape === otherHand.heldShape) {
        continue;
      }
      const score = grabScore(shape, hand.presencePoint.x, hand.presencePoint.y, TOUCH_RADIUS);
      if (score < closestScore) {
        closestScore = score;
        closestShape = shape;
      }
    }

    if (!closestShape) {
      return;
    }

    hand.heldShape = closestShape;
    hand.grabDx = hand.heldShape.offset.x - hand.presencePoint.x;
    hand.grabDy = hand.heldShape.offset.y - hand.presencePoint.y;
  }

  const frames = deltaTime / (1000 / 60);
  const followEase = 1 - Math.pow(1 - DRAG_FOLLOW_STRENGTH, frames);
  hand.heldShape.offset.x = lerp(
    hand.heldShape.offset.x,
    hand.presencePoint.x + hand.grabDx,
    followEase
  );
  hand.heldShape.offset.y = lerp(
    hand.heldShape.offset.y,
    hand.presencePoint.y + hand.grabDy,
    followEase
  );
}

function smoothStroke(points) {
  if (points.length < 2) {
    return points.map((point) => ({ ...point }));
  }

  let smoothed = resamplePoints(points, RESAMPLE_SPACING);
  for (let i = 0; i < SMOOTH_ITERATIONS; i++) {
    smoothed = chaikinSmooth(smoothed);
  }
  return smoothed;
}

function resamplePoints(points, spacing) {
  if (points.length < 2) {
    return points.map((point) => ({ ...point }));
  }

  const resampled = [{ ...points[0] }];
  let carry = 0;

  for (let i = 1; i < points.length; i++) {
    let prev = points[i - 1];
    let next = points[i];
    let segmentLength = dist(prev.x, prev.y, next.x, next.y);

    while (carry + segmentLength >= spacing) {
      const t = (spacing - carry) / segmentLength;
      resampled.push({
        x: lerp(prev.x, next.x, t),
        y: lerp(prev.y, next.y, t),
      });
      prev = resampled[resampled.length - 1];
      segmentLength = dist(prev.x, prev.y, next.x, next.y);
      carry = 0;
    }

    carry += segmentLength;
  }

  const lastOriginal = points[points.length - 1];
  const lastResampled = resampled[resampled.length - 1];
  if (dist(lastOriginal.x, lastOriginal.y, lastResampled.x, lastResampled.y) > spacing * 0.35) {
    resampled.push({ ...lastOriginal });
  }

  return resampled;
}

function chaikinSmooth(points) {
  if (points.length < 3) {
    return points.map((point) => ({ ...point }));
  }

  const smoothed = [{ ...points[0] }];

  for (let i = 0; i < points.length - 1; i++) {
    const p0 = points[i];
    const p1 = points[i + 1];
    smoothed.push({
      x: lerp(p0.x, p1.x, 0.25),
      y: lerp(p0.y, p1.y, 0.25),
    });
    smoothed.push({
      x: lerp(p0.x, p1.x, 0.75),
      y: lerp(p0.y, p1.y, 0.75),
    });
  }

  smoothed.push({ ...points[points.length - 1] });
  return smoothed;
}

function drawStrokePath(points) {
  if (points.length === 0) {
    return;
  }

  if (points.length === 1) {
    point(points[0].x, points[0].y);
    return;
  }

  if (points.length === 2) {
    line(points[0].x, points[0].y, points[1].x, points[1].y);
    return;
  }

  beginShape();
  curveVertex(points[0].x, points[0].y);
  for (const point of points) {
    curveVertex(point.x, point.y);
  }
  curveVertex(points[points.length - 1].x, points[points.length - 1].y);
  endShape();
}

// ---------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------

function drawShapes() {
  strokeCap(ROUND);
  strokeJoin(ROUND);
  noFill();

  for (const shape of shapes) {
    const held = shape === handA.heldShape || shape === handB.heldShape;
    const walking = shape.done && shape.walkAmount > 0.01;

    push();
    translate(shape.offset.x, shape.offset.y);
    for (const shapeStroke of shape.strokes) {
      const [r, g, b] = shapeStroke.color;
      // Light-drawn look: a soft halo in the stroke's color around a
      // core brightened toward white.
      drawingContext.shadowBlur = GLOW_BLUR;
      drawingContext.shadowColor = `rgba(${r}, ${g}, ${b}, 0.9)`;
      stroke(lerpColor(color(r, g, b), color(255), 0.35));
      strokeWeight(held ? 6 : 4);

      if (walking) {
        drawStrokePath(shapeStroke.points.map((point) => walkTransformPoint(shape, point)));
      } else {
        drawStrokePath(shapeStroke.points);
      }
    }
    pop();
  }

  drawingContext.shadowBlur = 0;
}

// Marker for each hand's presence point. Filled halo when the hand is
// open (can grab), collapsed hollow ring when it's a fist. Each hand
// carries its identity letter so A and B are unmistakable on screen.
function drawPresencePoint(hand) {
  if (!hand.presencePoint) {
    return;
  }

  const [r, g, b] = hand.markerColor;
  const { x, y } = hand.presencePoint;

  if (hand.isFist) {
    noFill();
    stroke(r, g, b, 160);
    strokeWeight(2);
    circle(x, y, 22);
    noStroke();
  } else {
    noStroke();
    fill(r, g, b, 50);
    circle(x, y, 46);
    fill(r, g, b, 200);
    circle(x, y, 10);
  }

  noStroke();
  fill(r, g, b, 220);
  textAlign(CENTER, CENTER);
  textSize(13);
  text(hand.id, x, y - 34);
}

// Fingertip cursor, tinted with the hand's current cap color; pinching
// tightens the ring and brightens the core.
function drawCursor(hand) {
  const cursor = hand.cursor;
  if (!cursor.initialized || cursor.alpha < 0.01) {
    return;
  }

  const cap = hand.capColor || DEFAULT_CAP;
  const base = color(cap.rgb[0], cap.rgb[1], cap.rgb[2]);
  const tint = lerpColor(base, color(255), cursor.pinchAmount * 0.5);
  const pulse = sin(millis() * 0.005) * 1.5;
  const ringSize = lerp(30, 18, cursor.pinchAmount) + pulse;
  const dotSize = lerp(8, 12, cursor.pinchAmount);

  noStroke();
  tint.setAlpha(40 * cursor.alpha);
  fill(tint);
  circle(cursor.x, cursor.y, ringSize * 1.8);

  noFill();
  tint.setAlpha(180 * cursor.alpha);
  stroke(tint);
  strokeWeight(2);
  circle(cursor.x, cursor.y, ringSize);

  noStroke();
  tint.setAlpha(255 * cursor.alpha);
  fill(tint);
  circle(cursor.x, cursor.y, dotSize);
}

// Debug panel (toggle with D): per hand, its identity, which physical
// hand it's riding on, the raw sampled fingertip color, and the
// classified cap.
function drawHandDebug() {
  if (!showHandDebug) {
    return;
  }

  const rowHeight = 34;
  const panelX = 16;
  let y = 16;

  textAlign(LEFT, CENTER);
  textSize(12);

  for (const hand of allHands) {
    const active = millis() - hand.lastSeenTime < HAND_LOST_GRACE_MS;
    const midY = y + rowHeight / 2;

    noStroke();
    fill(0, 0, 0, 130);
    rect(panelX, y, 330, rowHeight, 6);

    const [mr, mg, mb] = hand.markerColor;
    fill(mr, mg, mb);
    text(`Hand ${hand.id}`, panelX + 10, midY);

    if (!active) {
      fill(255, 110);
      text('not detected', panelX + 70, midY);
      y += rowHeight + 6;
      continue;
    }

    fill(255, 180);
    text(hand.handednessLabel ? `${hand.handednessLabel.toLowerCase()} hand` : 'tracking', panelX + 70, midY);

    // Raw sampled fingertip color.
    if (hand.sampledIndexRgb) {
      const [sr, sg, sb] = hand.sampledIndexRgb;
      fill(sr, sg, sb);
      stroke(255, 90);
      strokeWeight(1);
      rect(panelX + 212, midY - 9, 18, 18, 4);
      noStroke();
    }

    // Classified cap color and name.
    const cap = hand.capColor;
    if (cap) {
      fill(cap.rgb[0], cap.rgb[1], cap.rgb[2]);
      rect(panelX + 238, midY - 9, 18, 18, 4);
      text(cap.name, panelX + 264, midY);
    } else {
      fill(255, 110);
      text('no cap', panelX + 238, midY);
    }

    y += rowHeight + 6;
  }
}

function drawStatus() {
  noStroke();
  fill(255);
  textAlign(CENTER, CENTER);
  textSize(16);

  if (!capture?.loadedmetadata) {
    text('Starting camera...', width / 2, height / 2);
    return;
  }

  if (handTrackingError) {
    text('Hand tracking failed — camera works, drawing disabled.\nCheck the console.', width / 2, height / 2);
    return;
  }

  if (!handTrackingReady) {
    text('Loading hand tracking...', width / 2, height / 2);
  }
}

function keyPressed() {
  if (key === 'c' || key === 'C') {
    clearAllShapes();
  }

  if (key === 'd' || key === 'D') {
    showHandDebug = !showHandDebug;
  }
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
  if (capture) {
    capture.size(windowWidth, windowHeight);
  }
}
