# GNM Web Puppet

A browser-native rebuild of [`webcam_puppet/`](../webcam_puppet/README.md):
same idea — drive the GNM head from a webcam — but rendered on the GPU in a
web page, with no Python at runtime.

Built module by module, each independently runnable. This file tracks what is
done and what the next module is.

| # | Module | Status |
| --- | --- | --- |
| **M0** | Asset export + NumPy reference forward pass | **done** |
| **M1** | Web viewer and the stylized look | **done** |
| **M2** | GNM forward pass in WebGL | **done** |
| **M3** | Browser face tracking (MediaPipe Tasks Vision) | **done** |
| **M3.5** | Both views on one page, still unconnected | **done** |
| **M4a** | Landmark ↔ vertex correspondence, drawn in both views | **done** |
| **M4b** | Identity fit from tracked landmarks | **done** |
| **M4c** | Per-frame expression + head pose retarget | **done, prototype** |
| **M4e** | Corrective layer, driven by ARKit scores (cheek puff) | **done** |
| M4d | ARKit blendshapes for blink | next |
| M5 | Integration: identity capture flow, UI | — |

## Why a rewrite rather than a port

The desktop prototype works, but three of its properties do not survive
contact with the web, and two of them were limiting it anyway:

- **The renderer is a NumPy software rasterizer.** It costs 32 ms per frame —
  two-thirds of the frame budget — to produce flat Lambertian shading with one
  light, one albedo for all six mesh components, no shadows and no
  anti-aliasing. On a GPU that same frame is sub-millisecond, and the freed
  budget is what buys the look.
- **Expression is solved per frame by least squares.** MediaPipe's browser
  build already outputs 52 ARKit blendshape scores and a head-pose matrix.
  Consuming those turns the per-frame solve into a 52×K matrix-vector product
  and moves the hard fitting offline (M4), where it is testable — and it is
  the route to eye blinking, which landmark geometry never resolved.
- **Python at runtime** would mean a server between the camera and the screen.
  Everything here is static files.

## M0 — asset export

The shipped model is 150 MB of raw float32. Two bases dominate it and both are
mostly tail, so the export truncates them and stores them as float16:

```bash
# From the repository root.
python -m web_puppet.tools.export_assets --output_dir web_puppet/public/assets
```

| | kept | of | energy retained |
| --- | --- | --- | --- |
| identity | 64 | 253 | 98.9% |
| expression | 89 | 383 | ~99% within each region |

Expression components are budgeted **per region** — 24 left eye, 24 right eye,
32 lower face, 8 tongue, 1 pupil — never as a leading prefix of the whole
basis. The basis is region-blocked rather than variance-ordered, so a global
prefix selects eye components only and the resulting face cannot open its
mouth. This cost the prototype real time; see bug 1 in its README.

Output is `gnm_head.bin` (17.2 MB) plus `gnm_head.json`, a manifest of named
views into it: one fetch, and every array is a typed array over the buffer with
no parsing. Bases travel as raw float16 bits because that is exactly what a
WebGL half-float texture wants uploaded, so there is no decode on the critical
path.

Both files are generated and gitignored.

### What the export dropped, and why

`pose_correctives_regressor` is **entirely zero** in GNM head v3. The stage
that consumes it — regressing vertex offsets from joint rotations — therefore
contributes nothing at all. Dropping it removed 3.9 MB (18% of the download)
and one whole stage from the forward pass the WebGL shader has to implement.
`export_assets` raises if a future model populates the regressor, and
`test_pose_correctives_are_a_no_op` pins the fact so it cannot rot silently.

### Verifying

```bash
python -m web_puppet.tools.export_assets_test
```

`tools/reference_model.py` is a pure-NumPy forward pass that reads **only the
exported files** — it never imports `gnm.shape`. That makes it an executable
spec of what the WebGL implementation must reproduce: if the shader agrees
with it, and it agrees with `gnm.shape`, the shader is correct.

The test separates two error sources, because only one of them is a bug:

- **Fidelity** — the reference fed the kept coefficients versus `gnm.shape`
  fed the same coefficients scattered back into full-length vectors. Asserted
  tightly, and *relative* to the deformation the parameters actually produced,
  since float16 error scales with the magnitude it encodes and an absolute
  bound would pass or fail on the draw scale rather than on correctness.
  At float32 the two are **bit-exact**, so everything under the bound is
  quantization and nothing else. Worst observed: 3e-4 relative (0.013 mm on a
  43 mm deformation). A packing, stride or dtype error would be order 1.
- **Truncation** — measured and printed, not asserted tightly. Dropping
  components necessarily loses shape; that is the budget, not a defect. Over
  random draws across the full model: **~1 mm RMS, ~6 mm peak**.

## M1 — viewer and look

```bash
npm install          # first time only
npm run dev          # http://localhost:5173
```

A static head with sliders — no camera. Its job is to settle the visual
direction and to prove the browser forward pass drives the mesh, before
tracking exists to confuse the two.

### The look

Stylized rather than photoreal, and the reasoning is worth keeping: GNM ships
geometry and UVs but **no skin texture**, so realism would mean sourcing and
fitting a face scan and then living with the uncanny valley. Reading the head
as an object is honest about what the model is.

Four elements, in order of contribution:

1. **View-space shading, not world-space.** Key, fill and rim are all locked
   to the camera, so the head stays lit at any rotation with no environment
   map to download. This matters for M3: a tracked head swings through large
   angles and must never fall into shadow.
2. **A narrow Fresnel rim.** Nearly all silhouette definition comes from here.
3. **The quad wireframe** — the one element specific to GNM rather than
   generic styling, since it shows the model's real edge flow. Drawn from the
   exported `quad_edges` and sharing the surface's position buffer, so it
   cannot drift out of sync.
4. **Bloom and vignette**, thresholded high so only the rim and wire cross it.

### Brand palette

The render and the UI both use Edu's personal branding colours:

| | hex | hue | role in the render | role in the UI |
| --- | --- | --- | --- | --- |
| Teal | `#2f8871` | 164.5° | key light, rim, wireframe | accent, headings, sliders |
| Navy | `#283477` | 230.9° | ambient, shadow side, background | panel surface |
| Plum | `#612e65` | 295.6° | grazing fill on the turning edge | hover state |

A brand palette cannot be dropped into a lighting rig literally, and the two
failure modes are worth recording because they look like different problems:

- **Navy at its swatch value flooded everything.** Ambient is the one term
  that touches every pixel, so a mid-value navy lifted the whole head into a
  flat glow and bloom then spilled across the background.
- **Navy taken to near-black went flat the other way.** The shadow side lost
  its hue and the render read as flat jade instead of teal-on-navy.

It sits between the two, at `#1a2350`. The general rule: **hold the hue
exactly, vary the lightness.** Three mid-value swatches carry no tonal range,
and a light that never brightens past its own swatch cannot describe a form.

Two further tuning facts, both learned by getting them wrong first:

- **The broad surface has to stay dark.** A bright base plus rim plus bloom
  turns the whole head into a featureless lamp — the first version was
  unreadable for exactly this reason.
- **Rim strength must stay under 1.0** or the channels clip and a cyan rim
  renders white.

Per-component palettes are the other visible difference from the prototype,
which shaded all six components with one skin colour. Emissive eyes and
enamel teeth do more for the look than any amount of work on the skin.

### Verifying

```bash
npm test             # node --test, no browser needed
```

`src/gnm.ts` is the forward pass in TypeScript and imports **nothing** — no
Three.js, no DOM — precisely so it can be tested this way. It is checked
against golden vectors frozen from the Python reference:

```bash
python -m web_puppet.tools.dump_test_vectors
```

Chained end to end that gives `gnm.shape` ≡ `reference_model.py` ≡ `gnm.ts`.
Worst observed disagreement across five cases is **4e-4 mm** — float
accumulation order, nothing more.

The suite also pins the things that fail silently rather than loudly: that
posing actually moves the mesh (otherwise every case could agree by all being
the template), that component grouping partitions the index buffer, and that
normals are unit length and face the camera at the nose (a flipped winding
looks plausible until it doesn't).

### Cost

The forward pass runs on the **CPU** here, which is the right cost for
slider-driven shape:

| state | solve |
| --- | --- |
| neutral | 0.6 ms |
| all 64 identity components non-zero | 6.3 ms |

The gap is a deliberate skip of zero coefficients — each non-zero one costs a
full pass over 53k floats. That optimization stops paying in M3, where
expression changes every frame, which is what M2 is for.

## M2 — the forward pass on the GPU

Positions alone would fit in the render vertex shader. **Normals are why this
is a pipeline**: smooth normals need a vertex's neighbours, and a vertex
shader cannot see them. Evaluating each neighbour's basis inline would
multiply the work by the valence — up to 16 here — so positions are computed
once into a texture and a second pass gathers from it:

```
pass A   coefficients      -> position texture    (one texel per vertex)
pass B   position texture  -> normal texture      (gathers via adjacency)
pass C   the render, fetching both by vertex index
```

The mesh keeps a `position` attribute holding the template, purely so Three.js
can compute bounds; the shader ignores it and fetches the posed value by
index. Surface and wire read the same two textures, so they cannot desync.

`normal_adjacency` is the array M0 exports for pass B: per vertex, the
*cyclically following two* vertices of each incident triangle. That detail is
what makes the GPU normals exact rather than approximate — the cross product
of two edge vectors from any corner of a triangle in cyclic order is the same
vector, magnitude included, and that magnitude is twice the area and therefore
the weighting. `test_normal_adjacency_reproduces_face_normals` pins it.

### Cost

| | per frame |
| --- | --- |
| CPU, neutral | 0.6 ms |
| CPU, all identity components non-zero | 6.3 ms |
| **GPU dispatch** | **0.1–0.2 ms** |

Pass A evaluates all 153 components for every vertex unconditionally. The CPU
path skips zero coefficients, which is what made slider-dragging cheap — but
that optimisation stops paying the moment tracking drives every coefficient at
once, which is exactly the case M2 exists for.

### Verifying

There is no headless WebGL in the Node suite, so this check lives in the page:
**Verify vs CPU** reads the position target back and diffs it against
`gnm.ts`. Since `gnm.ts` is already checked against Python, agreeing here
extends the chain to the shader:

```
gnm.shape ≡ reference_model.py ≡ gnm.ts ≡ WebGL
```

Worst observed, over random identity across all 64 components plus several
expressions plus head pose: **2.1e-5 mm max, 5.0e-6 mm rms**.

One trap worth recording, because it produced a convincing 16 mm error that
looked like a shader bug: the check originally read back whatever the frame
loop had last rendered, so it compared the CPU's *current* answer against the
GPU's *previous* one. `verify` now drives the GPU from the same state it
evaluates the CPU with. A verification that does not control both sides of the
comparison is not a verification.

## M3 — tracking, on its own

```bash
npm run sync-assets   # first time only: vendors the MediaPipe runtime
npm run dev           # http://localhost:5173/tracker.html
```

The webcam with its landmarks drawn on it, and nothing else. **Not wired to
the GNM head**, deliberately: tracking and retargeting fail in different ways
— one as jittery or missing points, the other as a face that moves wrongly —
and debugging them together means never knowing which is at fault. The head
arrives in M4.

The page draws all 478 landmarks, the eye and iris contours, the face oval,
brows and lips, and a gaze ray per eye. Nothing starts until **Start camera**
is clicked.

### Eye direction

Gaze is read geometrically: the iris centre's offset from the midpoint of that
eye's two corners, divided by the eye's own half-width. Normalising by the eye
rather than by the image is what makes it survive the head moving towards or
away from the camera — offset and width scale together, so the ratio does not.
The vertical axis uses half the normaliser, because an eye is far wider than
it is tall and the raw ratio would otherwise read as enormous.

The readout also shows `eyeBlinkLeft` / `eyeBlinkRight` from the blendshape
classifier. Those are the route to blinking that the earlier prototype never
got working from eyelid geometry — see its README's "most promising next
step".

### What the tracker carries but does not yet use

`tracking.ts` asks the landmarker for all three of its outputs, because M4
needs the ones M3 does not draw:

| output | used in M3 | why it is requested now |
| --- | --- | --- |
| 478 landmarks | yes | drawn, and gaze is derived from them |
| 52 ARKit blendshapes | blink only, as text | the input to M4's retarget |
| 4x4 head pose | no | removes the prototype's per-frame pose solve |

### Vendored, not from a CDN

`npm run sync-assets` copies MediaPipe's WASM out of `node_modules` and the
`face_landmarker.task` bundle from the earlier prototype into `public/`. The
usual guidance is to point `FilesetResolver` at a CDN; vendoring keeps the app
working offline, pins the WASM to the same npm version the TypeScript types
came from — a mismatch there fails at runtime, not at build — and means no
third-party host sees a request from a page that is about to open a webcam.
Both are gitignored and regenerable.

## M3.5 — both at once, still unconnected

The viewer page gains a floating circular webcam view with the same tracking
overlay on it, started from **Camera → Start camera** in the panel. The head
is still slider-driven; **the tracker does not touch it**. This is a
compositing step, not a retarget.

Keeping it a separate step is what makes M4 diagnosable. Once the two are
wired together, a face that moves wrongly could be bad tracking or a bad
retarget, and there is no way to tell them apart while both are new. Watching
them side by side first establishes that the tracking half is sound.

`FacePip` owns the camera, the tracker and its own draw loop, and exposes
`latest` — the most recent `FaceFrame`. M4b consumes that, and needed one
addition: `aspect`, because MediaPipe normalizes x by width and y by height and
the fit has to undo that.

Two details worth keeping:

- **Drawing is shared with the tracker page** via `overlay.ts`. The only thing
  that differs between a full-frame view and a circular centre-cropped one is
  how normalized landmarks land in canvas pixels, so that mapping is the one
  parameter — `Projection` — and everything else is common.
- **A null detection is not a lost face.** `detect()` returns null whenever the
  timestamp has not advanced, which happens every time the render loop outruns
  the camera. The overlay keeps the previous frame rather than flickering.

## M4a — correspondence, shown in both views

**Correspondence (473)** in the panel draws the same points twice: on the
webcam image in the floating circle, and on the mesh itself. They are the 473
MediaPipe landmarks that have a GNM vertex.

The mapping is not derived here. `webcam_puppet/correspondence.py` built it by
rendering the head with a known camera, detecting landmarks on each render, and
back-projecting each onto the nearest visible skin vertex — exact up to
projected vertex spacing, median 1.4 px. `export_assets` reads the cached `.npz`
and folds it into the web assets; the flag `--correspondence_path` points at it,
and the export degrades gracefully if it is absent.

Calibration renders five poses, not one, and tests visibility rather than facing
direction. Both changes are about the mouth: on a closed-mouth render the upper-
and lower-lip landmarks land on the same vertex and half of every pair was
discarded, which cost 12 of the 40 lip landmarks, and a normal-only test let one
of the survivors map 68 mm inside the head. The lips are now fully corresponded
— 40 of 40, all 9 upper/lower pairs on distinct vertices. See the prototype's
README for the derivation.

This fixed what the correspondence overlay *showed* — mouth points sitting at or
behind the lip line — but measurement says it is not what limits mouth motion.
See *What actually limits the mouth* below.

### Colour is the whole mechanism

Drawing the points twice is useless unless you can tell *which* point is
which. Index-derived colours would not work, because neighbouring MediaPipe
indices are not neighbouring positions on a face. Instead hue follows
horizontal position and lightness follows vertical position, computed at
export time from the template vertex the landmark maps to. The colour is then
a readable function of where a point sits on a face, and both views index the
same array — so a point on your left cheek and its partner on the mesh's left
cheek are necessarily the same orange.

Two constraints this ran into:

- **Markers must not bloom.** The first version brightened each dot's centre;
  that pushed it past the bloom threshold, and a bloomed dot renders white,
  destroying the only information the colour carries. They are flat-shaded
  now, and the lightness ramp is capped well below white.
- **Markers must not z-fight.** A point sitting exactly on the surface it
  marks flickers against it, so the shader nudges each one about a
  half-millimetre towards the camera. Depth is still *tested*, so points on
  the far side of the head stay correctly hidden.

The markers read their position from the same texture the mesh does, so they
stay welded to the surface through every expression and pose with no per-frame
CPU work.

## M4b — identity from one frame

**Fit identity** in the panel takes the current tracked frame and writes the
identity coefficients. Expression and pose stay on the sliders, so a face that
looks wrong after a fit can only be the fit.

It is coarse by construction, and the numbers below say how coarse. 473 sparse
landmarks weakly constrain a 64-component basis: this moves proportions — face
width, chin, brow, nose projection — and is not a scan.

### The solve

Pose and shape need each other, so they alternate: a closed-form similarity fit
(Horn's quaternion method) for the pose, a ridge-regularized linear solve for
the shape, four times. Both halves are exact given the other. The normal matrix
is constant across iterations, so it is built and factored once and only the
right-hand side moves.

### The fit is differential, and that is the whole ballgame

Landmarks are compared against `correspondence_reference` — where the
landmarker places them on the *neutral* GNM head, cached from the calibration
render — not against the GNM vertices they map to. Those two clouds differ by:

| axis | rms | max |
| --- | --- | --- |
| x | 1.3 mm | 4.7 mm |
| y | 2.0 mm | 6.2 mm |
| **z** | **11.1 mm** | **67.5 mm** |

That is the landmarker's own idea of face shape, and it is systematic rather
than random. Fitted against the template it reads as a very deep face and lands
in the coefficients; fitted as displacement it cancels exactly, because it is
the same landmarker on both sides. Before this change the fit was *worse than
not fitting at all*. The prototype found the same thing for expression, which
is why its tracking is differential too.

This is the one thing M4b needed from offline tooling: `export_assets.py` now
carries `reference_landmarks` through into the web assets.

### Only the leading 24 components

The normal matrix' eigenvalues span 10⁵ on this point set — 15 components sit
above 1% of the top, 32 above 0.1% — so no single ridge both frees the head of
the basis and restrains its tail. The basis is variance-ordered, so truncation
is the honest cut. At 2 px landmark jitter, 64 components score 3.3 mm against
24's 2.4 mm.

### Rigid landmarks only

The solve runs on the 166 of 473 landmarks whose vertices carry the least
expression-basis displacement energy — the `rigid` mask, derived at export
time, not hand-picked. Whole-mesh rms against the face that generated the
frame:

| subject | rigid 166 | all 473 |
| --- | --- | --- |
| neutral | 2.4 mm | **1.1 mm** |
| smiling | **2.4 mm** | 4.8 mm |

All 473 is better on a face that is genuinely neutral and collapses on one that
is not, because a smile is absorbed as bone. Nobody holds a neutral face on
cue, so the flat line wins. `rigidOnly: false` is there if a capture flow ever
guarantees neutrality.

### What it actually scores

Over 40 random identities, drawn as **Random identity** draws them, with 2 px
of landmark jitter and depth error at 2% of face width:

| | fitted | template | better on |
| --- | --- | --- | --- |
| neutral subject | 3.75 mm | 5.08 mm | 29/40 faces |
| smiling subject | 3.78 mm | 5.08 mm | 29/40 faces |
| depth error at 4% | 8.59 mm | 5.08 mm | 6/40 faces |

So: about a quarter better than the template on three faces in four, unmoved by
expression — and it inverts if MediaPipe's depth is worse than assumed. That
last row is the one to watch on real faces; the synthetic benchmark cannot say
where real z jitter actually falls, only what happens either side of the line.

The coefficients themselves are **not** recoverable, and the tests deliberately
do not assert them: the near-null space is large enough that a coefficient can
be off by 0.6 while the fitted points agree to 0.1 mm. The face is the thing
that has to be right.

### Verifying

`npm test` fits synthetic faces whose answer is known — the frame is generated
by pushing the reference cloud through the identity basis and a known
similarity transform, then expressing it the way MediaPipe would. A neutral
subject fits as exactly neutral (0.0000 mm, peak |c| 0.0000), which is what
proves the differential base is right.

## M4c — the head follows the face

**Drive head** puts the webcam in charge of expression and head pose. This is
the first point at which the demo is the thing it was always meant to be.

It is a prototype and takes one shortcut, stated up front: it is **not** the
ARKit blendshape retarget the roadmap called for. Building a real 52 → 89
matrix means regressing GNM expression against ARKit's blend targets, and we do
not have those targets. What we do have is `fit.ts` and the prototype's proven
method, so M4c reuses both and the blendshape route stays open.

### The same machinery, pointed at the other basis

Align the tracked cloud onto the subject's own neutral face; whatever is left
over is expression. So:

1. Align on the **rigid** landmarks only. Aligning on all of them lets an open
   mouth drag the estimated head pose around with it.
2. The rotation that alignment removed **is** the head pose — inverted, since
   the fit maps camera to model.
3. Project the leftover displacement onto the expression basis at the
   corresponded vertices, ridge-regularized, over **all 473** landmarks. The
   mouth and brow are exactly the points that move, so excluding them the way
   the identity fit does would leave nothing to solve.

The normal matrix depends on the subject, not the frame, so it is built once
per identity; each frame costs one right-hand side and one back-substitution.

### Identity first, or the face wears a permanent expression

`setIdentity` rebuilds the neutral cloud expressions are measured from, and the
viewer calls it whenever **Fit identity** runs. Skipping it is not subtle — in
the test that guards it, peak expression goes from 0.000 to 3.000, the clamp,
because the subject's own bone structure is being read as a grimace.

### Measured on synthetic frames

| | result |
| --- | --- |
| resting face | 0.0000 mm rms, no invented expression or rotation |
| mouth expression | 0.17 mm rms, coefficients within 0.15 |
| head turned 0.35 rad | solved `[-0.000, 0.350, -0.000]`, no leak into expression |

Coefficients come back slightly *short* of the truth, which is ridge shrinkage
working as intended: a demo that under-reacts beats one that twitches.

### Known-imperfect, deliberately

- **Blink barely works.** Eyelid geometry is what defeated the Python
  prototype, and nothing here fixes it. MediaPipe's 52 ARKit scores are the
  answer and `FaceFrame` already carries them — that is M4d.
- **Smoothing is a flat EMA at 0.45**, so fast motion lags. Real numbers on a
  real face have not been taken; these are synthetic frames only.
- **The retarget runs on a 33 ms timer**, not in the render loop, because the
  camera produces 30 fps against the page's 60 and solving per rendered frame
  would do half its work twice.

### What actually limits the mouth

The reported symptom was that opening your mouth inflates the model's lips
instead of parting them. It is not a correspondence problem and not a gain
problem. **The mouth solve was over-parameterized**, and the excess components
went into fitting landmark noise.

Rendering a pure jaw-open at three amplitudes and solving it back:

| true aperture | solved, 32 mouth components | solved, 4 |
|---|---|---|
| +2.5 mm | −2.9 mm | +1.8 mm |
| +5.0 mm | −3.7 mm | +3.8 mm |
| +7.4 mm | −3.5 mm | +5.4 mm |

At 32 the aperture comes back **negative** — the model's mouth closes as the
subject's opens — and it gets there by pinning four components at the ±3 clamp
while never touching `lower_face_region_000`, the component that actually opens
the jaw. A wrong shape at saturation amplitude is what reads as inflated lips.
At 4, the sign is right, the amplitude is ~73% of truth, and nothing clamps.

The trailing components of a region are fine detail. Sparse, noisy landmarks
cannot constrain them, so least squares spends them on noise, in combinations
large enough to swamp the gross shape. The fix is `RetargetOptions.regionBudget`,
defaulting to 4 for `lower_face_region`. It restricts only the per-frame solve —
the full 32 stay in the model, so sliders and the identity fit are unchanged.

Raising the ridge instead also fixes the sign (0.3 and above), but tops out
around 48% of the true aperture against 73% for the budget, and it blunts every
region at once rather than the one that needs it.

Two things measured and rejected as the cause:

- **The correspondence.** Repairing the lip collisions took coverage from 28 of
  40 landmarks to 40 of 40, and moved recovery not at all — 56% before, 50%
  after, inside the spread of a 12-pose benchmark. Worth doing on its own terms;
  it was not this bug.
- **A gain knob**, *as measured against the broken solve*. With 32 components
  the per-pose optimal gain had mean 1.30 and sd 0.77, ranging −0.17 to 2.43,
  and a fixed 1.30 changed nothing. That instability was the sign inversion, not
  the data — see below, where the same measurement on the fixed solve reverses
  the conclusion.

### The remaining shortfall is a scale factor, and is corrected

With the budget in place the mouth opens the right way but short — about
three-quarters of life-size. Measured on pure jaw-open at five amplitudes:

| observations | aperture returned |
|---|---|
| the model's own vertices | 100% of truth, at every amplitude |
| MediaPipe's landmarks | 72–77%, across a 3× range of opening |

Ridge is not the cause: sweeping 0.001 to 0.05 does not move the result by
0.1 mm. The deficit is in the landmarks and it is proportional, so `regionGain`
corrects it — 1.35 for `lower_face_region`. The per-case optimal gain is 1.29 to
1.39 (mean 1.35, sd 0.04); at a fixed 1.35 the worst aperture error across the
range is 0.22 mm and nothing reaches the clamp.

Note the shape of that argument. A gain is only honest because the shortfall was
first shown to be a *constant ratio* — before the budget fix it was not, and
applying one would have been a fudge over a solver bug. Only the mouth is
corrected; the eye regions have no equivalent measurement yet.

Caveat: these are renders of the untextured GNM head, and MediaPipe places lip
landmarks from vermilion contrast on real faces, so the measurement may
understate real-webcam quality.

```
web_puppet/
  tools/
    export_assets.py       npz -> gnm_head.bin + gnm_head.json
    reference_model.py     pure-NumPy forward pass over the exported assets
    export_assets_test.py  reference vs gnm.shape, plus topology invariants
    dump_test_vectors.py   freezes reference output for the TypeScript test
  src/
    gnm.ts                 CPU forward pass; imports nothing, runs under node
    gnm.test.ts            gnm.ts vs the frozen Python vectors
    fit.ts                 identity from one frame; imports nothing either
    fit.test.ts            round trips synthetic faces through the solver
    retarget.ts            per-frame expression + head pose, on fit.ts
    retarget.test.ts       round trips expressions and head turns
    gpu.ts                 the same forward pass in WebGL, two passes
    look.ts                materials, scene, post-processing
    main.ts                viewer page: sliders, GPU verification, the pip
    tracking.ts            MediaPipe wrapper; landmarks, blendshapes, pose
    overlay.ts             landmark drawing, shared by both views
    tracker.ts             tracker page: full-frame webcam overlay
    pip.ts                 the floating circular tracker in the viewer
  public/assets/           generated, gitignored
  testdata/                generated, gitignored
```
