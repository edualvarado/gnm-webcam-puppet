# GNM Webcam Puppet

Drives the [GNM Head](../gnm/shape/README.md) parametric model from a webcam in
real time. Your face is tracked with MediaPipe, the landmarks are solved into GNM
identity / expression / pose parameters, and the resulting mesh is rendered beside
the camera preview at interactive rates.

This is a demo built on top of the GNM release, not part of the upstream package —
it lives outside `gnm/` so it never collides with it.

## Quick start

```bash
# From the repository root.
pip install -e "./gnm/shape"           # the GNM model itself
pip install -r webcam_puppet/requirements.txt

python -m webcam_puppet.app
```

The face landmarker model (~3.8 MB) downloads to `assets/` on first run.

| Key | Action |
| --- | --- |
| `n` | Capture identity from the current frame — hold a neutral face first |
| `r` | Reset identity back to the GNM template |
| `m` | Toggle mirroring of the camera preview |
| `q` | Quit |

The window is resizable and keeps its aspect ratio; dragging it scales the view
without re-rendering. `--render_size` (default 720) is the separate question of
how much detail is actually drawn, and is the main cost in the frame — see
*Per-frame cost*.

Useful flags: `--camera N` to pick a device, `--render_size`, `--smoothing`,
`--expression_gain`, `--record out.mp4` to capture the composited view.

### If the webcam does not appear

```bash
python -m webcam_puppet.app --list_cameras
```

```
index  liveness  assessment
    0      0.11  delivers frames but looks blank (infrared or virtual device?)
    1     25.45  looks like a real webcam

Auto-selection would use --camera 1.
```

Then `--camera 1` if you want to pin it.

`--camera` is auto by default, and auto-selection is deliberately more careful
than picking the first device that opens, because three separate things go wrong:

- **`isOpened()` proves nothing.** A device can open and then fail every read.
- **Cameras need warm-up.** The first read or two after opening can fail on
  hardware that works fine 200 ms later, so one failed read is not a verdict.
- **Delivering frames is not showing a picture.** This is the one that actually
  bit: on the development laptop the Windows Hello **infrared sensor enumerates
  as index 0**, ahead of the real webcam, and returns frames that are nearly
  black, pure greyscale and completely static. Any check of the form "did a frame
  arrive" selects it, and the app appears to have no webcam.

So candidates are scored on how *alive* they look — colour content plus
frame-to-frame change. The infrared device scores 0.11; the webcam beside it
scores ~25. Scoring and selection happen in one pass, keeping the winning device
open: an earlier version scored everything, released it all, then re-opened the
winner, and that re-open could transiently fail and silently fall through to the
blank infrared device.

### Verifying without a webcam

`selftest.py` poses GNM with known parameters, renders those poses as stand-in
camera frames, runs the real pipeline over them, and checks the recovered
parameters against the truth. This is how to validate a change on a machine with
no camera:

```bash
python -m webcam_puppet.selftest --output_dir /tmp/puppet_selftest
```

### Tests

```bash
python -m webcam_puppet.renderer_test
python -m webcam_puppet.solver_test
python -m webcam_puppet.camera_source_test
```

Run them as modules from the repository root; `webcam_puppet` is imported by
package name and is not pip-installed.

## How it works

```
webcam frame
  -> tracker.py       MediaPipe FaceLandmarker -> 478 landmarks
  -> solver.py        landmarks -> GNM identity / expression / pose
  -> gnm.shape        parameters -> 17.8k mesh vertices
  -> renderer.py      mesh -> shaded image (~32 ms at 720x720)
```

| Module | Role |
| --- | --- |
| `correspondence.py` | Derives which GNM vertex each MediaPipe landmark corresponds to |
| `geometry.py` | Axis conventions and the similarity (Umeyama) fit |
| `solver.py` | Per-frame expression and pose solving |
| `renderer.py` | Vectorized scanline rasterizer with an exact z-buffer |
| `tracker.py` | MediaPipe wrapper and model download |
| `camera_source.py` | Selecting a camera that actually shows a picture |
| `selftest.py` | Headless end-to-end verification |

### Rasterizing 35k triangles from Python

Splatting each vertex as a small square is the obvious way to keep a Python
renderer vectorized, and it is what this started as, but it draws a shell of
dots rather than a surface — at any size worth looking at, the head is visibly
made of tiles. Drawing real triangles instead means never touching one at a
time: each triangle is expanded directly into its **scanline spans**, so the
per-pixel work is one fused pass over exactly the pixels the mesh covers, with
no Python loop over primitives and no samples tested and thrown away.

Two details carry most of the speed. Shading interpolates a scalar light
intensity rather than RGB, with albedo and gamma resolved through a 256-entry
lookup table at the very end — a third of the interpolation traffic, and the
gamma curve is evaluated 256 times per session instead of a million times per
frame. And visibility is resolved on 1/z, which unlike z is linear in screen
space, so the z-buffer needs no per-pixel division.

### Correspondence is derived, not hand-authored

MediaPipe's face mesh and GNM use unrelated topologies. The usual approach is to
hand-transcribe landmark index tables, which is error-prone and hard to verify.
Instead the mapping is derived by construction: render the GNM head with a known
camera, detect landmarks on that render, and back-project each one onto the
nearest visible GNM skin vertex. Because the camera is known, this is exact up to
the projected vertex spacing. All 478 landmarks match, none further than 6.4 px
from a candidate vertex, median 1.4 px. The result is cached in
`assets/correspondence.npz`.

Calibration is **multi-shot**, and on the mouth that is the difference between
working and not. A neutral render cannot separate the lips: with the mouth
closed, MediaPipe's upper- and lower-lip landmarks project onto the same GNM
vertex, and the one-landmark-per-vertex rule then discards half of every pair.
Single-shot kept 28 of the 40 lip landmarks and left 8 of the 9 upper/lower pairs
sharing one vertex — the lower lip was, in effect, not tracked at all. Rendering
four extra poses (the two leading mouth expression components at both signs,
amplitude 1.5) and merging the matches recovers all 40 and all 9, for 473 unique
vertices.

The merge takes the neutral pose first and falls through to the posed ones only
for landmarks neutral could not place. That ordering is load-bearing: a deformed
render slides the surface under the landmarks, so a posed match can be nearer in
pixels while naming the wrong vertex, and the vertex is what selects the
expression-basis rows the solver fits through. Ranking by distance across all
poses instead cost a quarter of the recovered mouth motion when measured.

Candidates are also tested for **visibility, not just facing direction**. A
vertex whose normal points at the camera need not be visible — the far wall of an
open mouth faces the camera too — and matching a lip landmark to one puts the
correspondence inside the head. Without the test, one upper-lip landmark mapped
to a vertex 68 mm deep, and 16 of 28 lip points sat behind the visible surface.

The rigid subset used for pose estimation is derived from the model too, rather
than being hand-picked: candidate vertices must be (a) skinned at least 90% to the
head joint and (b) among the lowest-energy vertices in the expression basis. Both
conditions matter — see *Bugs worth knowing about* below.

### Tracking is differential

The solver fits how far landmarks have moved from their neutral positions, not
where they are absolutely. This is not a refinement; absolute fitting does not
work here. MediaPipe reports its own idea of face shape with a `z` that is not
metric, so its landmarks never coincide with the GNM vertices they map to — even
on the very render they were detected from. Fitting absolute positions therefore
asks GNM to deform into MediaPipe's face, which registered **8.9 mm** of fit error
and **5.9 mm** of invented expression on a resting face. Fitting displacements
from a neutral reference cancels that constant bias, bringing a resting face to
0.6 mm.

### Per-frame cost

Both bases are pre-factored once at startup via GNM's own
`fitting_utils.PCABasisProjection`, so each frame is one similarity fit plus one
matrix product. Measured over live webcam frames at the default `--render_size`:

| Stage | ms |
| --- | --- |
| render | 32 |
| GNM forward pass | 8 |
| MediaPipe | 5 |
| solve + composite | 3 |
| **total** | **~48 (21 fps)** |

Rendering dominates and scales with the pixels the head covers, so
`--render_size` is the dial: 640 gives ~24 fps and 560 ~27 fps.

## Measured behaviour

From `selftest.py`, driving known GNM poses through the full pipeline:

| Case | Deformation | Recovered | Yaw truth → fit |
| --- | --- | --- | --- |
| neutral | — | — (0.08 mm residual) | 0° → 0.6° |
| yaw 20° | — | — (1.56 mm residual) | 20° → 25.0° ✗ |
| mouth | 2.13 mm | 17% | 0° → 2.5° |
| mouth + yaw | 5.30 mm | 32% | 15° → 14.1° |
| eyes | 0.78 mm | not recovered | 0° → 0.1° |
| combined | 3.21 mm | 23% | −12° → −14.9° |

**Expression is directionally right but muted.** The ceiling is the landmarker,
not the solver: solving the same deformation from *exact* landmarks recovers
~87%, but MediaPipe's per-landmark displacement has only weakly agreeing
direction (per-axis correlation 0.73 / 0.49 / 0.47 for x / y / z; mean cosine
0.37) despite roughly correct magnitude (0.89×).

**Head pose is usually within ~2°, but the 20° yaw case reads 25° and fails its
check.** This appeared when the stand-in frames moved from splats to a solid
mesh, and it is worth knowing that the old passing number was measured under
favourable conditions: the correspondence and the stand-in frames were rendered
by the same splatter, so both carried the same ~3 px dilated silhouette, and
pose is read off exactly that silhouette. Re-deriving the correspondence from
the mesh render but keeping splat-rendered frames puts yaw at 21.3° — so the
frames are what moved, not the correspondence. Reading it the other way: solid
frames are the more honest stand-in for a webcam, and under them yaw is less
accurate than previously claimed. Open item.

`--expression_gain` (default 2.0) amplifies expression so motion reads on screen.
It is exaggeration, not accuracy — it scales direction error equally, and past
about 3 the face turns visibly lumpy. `landmark_rmse` is always reported for the
unscaled fit.

### Known limitations

- **Eyelids and blinking are not tracked.** The deformation is sub-millimetre and
  MediaPipe's eyelid landmark geometry does not resolve it. See below for the fix.
- **Expression and pose are not fully separable.** The neutral reference is
  captured frontal, so as the head turns, pose-dependent landmark drift leaks into
  expression: ~0.6 mm frontal, ~1.8 mm at 20° yaw.
- **Beyond ~30° yaw** the far cheek self-occludes and its landmarks are
  extrapolated, so pose degrades.
- **Identity capture (`n`) is coarse.** 468 sparse landmarks weakly constrain 253
  identity components; it shifts the face noticeably but is not a scan.
- **Tongue and pupil components are never solved** — neither is observable from
  outside the face, so fitting them would only absorb noise.

### Most promising next step

Drive the eyes from MediaPipe's **blendshape scores** rather than landmark
geometry. The landmarker already outputs 52 ARKit-style scores including
`eyeBlinkLeft` / `eyeBlinkRight`, which are trained classifiers and far more
reliable than eyelid landmark positions. `tracker.py` can already request them
(`output_blendshapes=True`). What is missing is a mapping from blendshape scores to
GNM eye-region coefficients — obtainable by sweeping GNM eye components, rendering
each, reading back the blendshape scores, and fitting the inverse.

## Bugs this went through

Recorded because each was invisible to the obvious test and cost real time.

1. **Expression components are region-blocked, not variance-ordered.** The basis
   concatenates left eye (0–99), right eye (100–199), lower face (200–349), tongue
   (350–381), pupils (382). Taking "the leading 80 components" — the natural
   PCA-style default — selects eye components *only*, so the puppet could not open
   its mouth. Components are now budgeted per region.
2. **A single ridge weight cannot serve both regions.** Eye and lower-face blocks
   differ ~100× in displacement energy, so a ridge tuned for the mouth flattened
   eyelids entirely: at 2e-4 a known eye expression reconstructed *worse* than
   predicting neutral. Now 1e-5.
3. **Rigid anchors need a skinning constraint, not just low expression energy.**
   Neck vertices have zero expression energy, so they looked maximally rigid — but
   they do not move when the head joint rotates. Including them made the
   similarity fit average moving and stationary points, under-estimating 15° of yaw
   as 8.3°. Candidates are now required to be ≥90% skinned to the head joint.
4. **"Delivers a frame" is not "shows a picture."** The infrared sensor at index
   0 passed every readiness check while displaying nothing, so the app looked
   like it had no webcam. Selection now scores colour and motion. Its first fix
   was also wrong: scoring released every device before re-opening the winner,
   and a transient re-open failure fell through to the blank device.
5. **The z-buffer must match the incoming depth dtype.** Visible fragments are
   identified by exact equality against the buffer, so a narrower buffer rounds
   every value on write and matches almost nothing — this dropped 98% of
   fragments. `template_vertex_positions` is `float32` while the GNM forward
   pass returns `float64`, so every smoke test on the template passed while
   posed meshes rendered as confetti. The rasterizer now casts depth to
   `float32` explicitly rather than inheriting whatever arrives.
6. **Edge functions cancel catastrophically in absolute pixel coordinates.**
   Written as `a*x + b*y + c`, the constant is a product of two ~500 px terms
   while the value it produces is order 1. In `float32` the rounding error
   exceeds the answer near an edge, and a pixel on a shared edge ends up
   claimed by neither triangle: isolated dark specks along the lips and nose,
   showing the inside of the mouth through the crack. Evaluating in
   bounding-box-local coordinates keeps every term the size of one triangle.
7. **Winding sign has to be applied to the barycentric divisor too.** Front
   faces come out clockwise on screen, so the edge functions are negated to
   make "inside" positive — but the signed area they are divided by is still
   negative. Negating only the edges yields barycentric weights that are
   correct in magnitude and uniformly negative, which sails through the inside
   test and then loses every fragment to a z-buffer that only accepts positive
   depths. The image comes out empty with no error anywhere.
