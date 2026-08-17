/**
 * The stylized look: materials, scene and post-processing.
 *
 * Deliberately non-photoreal. GNM ships geometry, UVs and no skin texture, so
 * chasing realism would mean sourcing and fitting a face scan and then living
 * with the uncanny valley. Reading the head as an *object* -- a lit surface
 * under a glowing wireframe -- is honest about what the model is, and it is
 * the version that survives a screenshot.
 *
 * Four things carry the look, in rough order of contribution:
 *
 *   1. A view-space normal gradient instead of a light-driven albedo. The
 *      surface tone is a function of which way it faces the camera, which is
 *      the matcap trick: it reads as sculpted under any rotation without an
 *      environment map to download.
 *   2. A strong Fresnel rim. Almost all the silhouette definition comes from
 *      here, and it is what makes the head separate from a black background.
 *   3. The quad wireframe, drawn additively so bloom picks it up. This is the
 *      one element that is specific to GNM rather than generic styling -- it
 *      shows the model's actual edge flow.
 *   4. Bloom plus vignette, which glue the first three together.
 */

import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

import type { GnmModel } from './gnm.ts';
import { GpuEvaluator } from './gpu.ts';

/** How one mesh component is shaded. */
export interface Palette {
  deep: number;
  lit: number;
  fill: number;
  fillStrength: number;
  rim: number;
  rimPower: number;
  rimStrength: number;
  specular: number;
  emissive: number;
  emissiveStrength: number;
}

/**
 * Per-component palettes, keyed by GNM's `mesh_component_names` order:
 * skin, left_eye, right_eye, upper_teeth_and_gums, lower_teeth_and_gums,
 * tongue.
 *
 * Giving the eyes their own emissive palette is the single highest-value
 * difference from the old prototype, which shaded all six components with one
 * flat skin colour -- matte skin-coloured eyeballs read as dead more than any
 * other single error.
 */
/**
 * Brand palette. Every colour below is one of these three hues -- only the
 * value moves.
 *
 *   teal #2f8871   hue 164.5deg
 *   navy #283477   hue 230.9deg
 *   plum #612e65   hue 295.6deg
 *
 * A brand palette cannot be used literally as a lighting rig: three mid-value
 * colours give no range, and a light that never brightens past its swatch
 * cannot carry form. Holding the hue exactly while letting lightness vary is
 * what keeps the render recognisably on-brand and still readable, so every
 * derived tone here is the same hue at a different lightness.
 */
export const BRAND = {
  teal: 0x2f8871,
  tealBright: 0x54c4a7,
  tealPale: 0x7cdec5,
  navy: 0x283477,
  // Navy at its swatch value is far too bright to serve as ambient: it is the
  // one term that touches every pixel, so using it literally lifted the whole
  // head into a flat glow and bloom then spilled across the background. Taken
  // all the way down, though, the shadow side goes neutral and the render
  // reads as flat jade rather than teal-on-navy. This sits between: navy is
  // legible in shadow without lifting the midtones.
  navyDim: 0x1a2350,
  navyDeep: 0x0b0f24,
  plum: 0x612e65,
  plumBright: 0x9b3ca1,
} as const;

/** The void the head sits in: navy, taken almost to black. */
export const BACKGROUND = 0x070a18;

export const COMPONENT_PALETTES: Palette[] = [
  // skin -- navy shadow, teal key, plum graze. The broad surface stays dark
  // on purpose: everything bright in the frame should be the rim, the wire or
  // a highlight, or bloom smears the silhouette into a lamp and the form
  // disappears.
  {
    deep: BRAND.navyDim, lit: BRAND.teal, fill: BRAND.plumBright,
    fillStrength: 0.85,
    // Kept under 1.0 so the rim keeps its hue: past that the channels clip
    // and a teal rim renders as a white one.
    rim: BRAND.tealBright, rimPower: 3.4, rimStrength: 0.95, specular: 0.35,
    emissive: 0x000000, emissiveStrength: 0,
  },
  // left_eye
  {
    deep: BRAND.navyDeep, lit: BRAND.tealBright, fill: BRAND.plumBright,
    fillStrength: 0.20,
    rim: BRAND.tealPale, rimPower: 2.0, rimStrength: 1.7, specular: 1.6,
    emissive: BRAND.teal, emissiveStrength: 0.30,
  },
  // right_eye
  {
    deep: BRAND.navyDeep, lit: BRAND.tealBright, fill: BRAND.plumBright,
    fillStrength: 0.20,
    rim: BRAND.tealPale, rimPower: 2.0, rimStrength: 1.7, specular: 1.6,
    emissive: BRAND.teal, emissiveStrength: 0.30,
  },
  // upper_teeth_and_gums
  {
    deep: BRAND.navyDim, lit: 0xa8c8bd, fill: BRAND.plumBright, fillStrength: 0.12,
    rim: 0xcfeee2, rimPower: 3.6, rimStrength: 0.65, specular: 0.5,
    emissive: 0x000000, emissiveStrength: 0,
  },
  // lower_teeth_and_gums
  {
    deep: BRAND.navyDim, lit: 0xa8c8bd, fill: BRAND.plumBright, fillStrength: 0.12,
    rim: 0xcfeee2, rimPower: 3.6, rimStrength: 0.65, specular: 0.5,
    emissive: 0x000000, emissiveStrength: 0,
  },
  // tongue -- the one component where plum leads rather than accents.
  {
    deep: 0x1a0a26, lit: 0x8f3d84, fill: 0xbf5fc4, fillStrength: 0.45,
    rim: 0xc06fc4, rimPower: 2.6, rimStrength: 0.85, specular: 0.7,
    emissive: 0x000000, emissiveStrength: 0,
  },
];

/**
 * Shared by the surface and the wire.
 *
 * Neither reads the mesh's own position or normal attributes: both are
 * fetched from the textures the GPU passes wrote, indexed by vertex id. The
 * `position` attribute still exists, holding the template, purely so Three.js
 * has something to compute bounds from.
 */
const SURFACE_VERTEX_SHADER = /* glsl */ `
  uniform sampler2D uPositions;
  uniform sampler2D uNormals;
  uniform vec2 uVertexSize;

  attribute float aIndex;

  varying vec3 vViewNormal;
  varying vec3 vViewPosition;

  void main() {
    vec2 uv = (vec2(
      mod(aIndex, uVertexSize.x),
      floor(aIndex / uVertexSize.x)
    ) + 0.5) / uVertexSize;

    vec3 posed = texture2D(uPositions, uv).xyz;
    vec3 posedNormal = texture2D(uNormals, uv).xyz;

    vViewNormal = normalize(normalMatrix * posedNormal);
    vec4 viewPosition = modelViewMatrix * vec4(posed, 1.0);
    // Pointing from the surface towards the eye, which is where the camera
    // sits in view space, so the Fresnel term needs no camera uniform.
    vViewPosition = -viewPosition.xyz;
    gl_Position = projectionMatrix * viewPosition;
  }
`;

const SURFACE_FRAGMENT_SHADER = /* glsl */ `
  uniform vec3 uDeep;
  uniform vec3 uLit;
  uniform vec3 uFill;
  uniform float uFillStrength;
  uniform vec3 uRim;
  uniform float uRimPower;
  uniform float uRimStrength;
  uniform float uSpecular;
  uniform vec3 uEmissive;
  uniform float uEmissiveStrength;

  varying vec3 vViewNormal;
  varying vec3 vViewPosition;

  // Everything is evaluated in view space, so the whole rig -- key, fill and
  // rim -- is locked to the camera rather than to the model. The head can be
  // rotated to any angle and the lighting still reads, which is what the
  // eventual live tracking needs.
  const vec3 KEY_DIRECTION = vec3(-0.55, 0.42, 0.72);
  // Behind and to the right, so the fill grazes the turning edge instead of
  // washing across the cheek -- a frontal fill reads as a stain on the skin
  // rather than as a second light.
  const vec3 FILL_DIRECTION = vec3(0.86, -0.16, -0.30);

  void main() {
    vec3 normal = normalize(vViewNormal);
    vec3 toEye = normalize(vViewPosition);

    // Hemispheric ambient, near-black underneath. This is the only term that
    // touches every pixel, so it is kept very dark on purpose.
    float sky = normal.y * 0.5 + 0.5;
    vec3 color = mix(uDeep * 0.4, uDeep, sky);

    // Cool key from the upper left carries the form.
    vec3 key = normalize(KEY_DIRECTION);
    // The exponent is falloff, not brightness: a sharper curve keeps the
    // shadow side genuinely dark so the navy ambient and the plum graze have
    // somewhere to read.
    float diffuse = clamp(dot(normal, key), 0.0, 1.0);
    color += uLit * pow(diffuse, 1.9);

    // Warm fill from the opposite side. The hue opposition is what stops a
    // single-colour render reading as flat, and it costs one more dot product
    // rather than a second render pass.
    float fill = clamp(dot(normal, normalize(FILL_DIRECTION)), 0.0, 1.0);
    color += uFill * pow(fill, 2.3) * uFillStrength;

    // Tight Blinn highlight, just enough to imply a surface rather than a
    // matte solid.
    vec3 halfway = normalize(key + toEye);
    color += uLit * uSpecular * pow(clamp(dot(normal, halfway), 0.0, 1.0), 48.0);

    // The narrow bright rim that defines the silhouette, and the only part of
    // the surface intended to cross the bloom threshold.
    float fresnel = pow(1.0 - clamp(dot(normal, toEye), 0.0, 1.0), uRimPower);
    color += uRim * fresnel * uRimStrength;

    color += uEmissive * uEmissiveStrength;

    gl_FragColor = vec4(color, 1.0);
  }
`;

const WIREFRAME_FRAGMENT_SHADER = /* glsl */ `
  uniform vec3 uColor;
  uniform float uOpacity;

  varying vec3 vViewNormal;
  varying vec3 vViewPosition;

  void main() {
    // Fading the wire towards the silhouette rather than drawing it flat
    // keeps the topology readable across the face while still concentrating
    // brightness where the form turns away -- which is what bloom then picks
    // up as a rim.
    vec3 normal = normalize(vViewNormal);
    vec3 toEye = normalize(vViewPosition);
    float facing = clamp(dot(normal, toEye), 0.0, 1.0);
    float weight = mix(0.35, 1.0, pow(1.0 - facing, 1.6));
    gl_FragColor = vec4(uColor * weight, uOpacity * weight);
  }
`;

/**
 * Markers for the vertices MediaPipe landmarks map onto.
 *
 * They read their position from the same texture the mesh does, so they stay
 * welded to the surface through every expression and pose without any
 * per-frame work on the CPU.
 */
const CORRESPONDENCE_VERTEX_SHADER = /* glsl */ `
  uniform sampler2D uPositions;
  uniform vec2 uVertexSize;
  uniform float uPointSize;

  attribute float aIndex;
  attribute vec3 aColor;

  varying vec3 vColor;

  void main() {
    vec2 uv = (vec2(
      mod(aIndex, uVertexSize.x),
      floor(aIndex / uVertexSize.x)
    ) + 0.5) / uVertexSize;

    vColor = aColor;
    vec4 viewPosition = modelViewMatrix * vec4(texture2D(uPositions, uv).xyz, 1.0);
    // A marker sitting exactly on the surface it marks z-fights with it, so
    // nudge it towards the camera by well under a millimetre.
    viewPosition.z += 0.0015;

    gl_PointSize = uPointSize / max(-viewPosition.z, 1e-4);
    gl_Position = projectionMatrix * viewPosition;
  }
`;

const CORRESPONDENCE_FRAGMENT_SHADER = /* glsl */ `
  varying vec3 vColor;

  void main() {
    vec2 offset = gl_PointCoord - 0.5;
    float radius = length(offset);
    if (radius > 0.5) discard;
    // Flat colour, deliberately. Brightening the centre pushed these past the
    // bloom threshold, and a bloomed dot loses its hue to white -- which
    // destroys the only thing these markers are for.
    float alpha = smoothstep(0.5, 0.32, radius);
    gl_FragColor = vec4(vColor, alpha);
  }
`;

/** Vignette and grain, applied after tone mapping. */
const FINISH_SHADER = {
  uniforms: {
    tDiffuse: { value: null as THREE.Texture | null },
    uTime: { value: 0 },
    uVignette: { value: 1.05 },
    uGrain: { value: 0.035 },
  },
  vertexShader: /* glsl */ `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: /* glsl */ `
    uniform sampler2D tDiffuse;
    uniform float uTime;
    uniform float uVignette;
    uniform float uGrain;
    varying vec2 vUv;

    void main() {
      vec4 color = texture2D(tDiffuse, vUv);

      vec2 offset = vUv - 0.5;
      float vignette = 1.0 - uVignette * dot(offset, offset);
      color.rgb *= clamp(vignette, 0.0, 1.0);

      // Cheap hash grain. Without it the large flat background areas band
      // visibly once bloom has smeared a gradient across them.
      float noise = fract(sin(dot(vUv * 1024.0 + uTime, vec2(12.9898, 78.233))) * 43758.5453);
      color.rgb += (noise - 0.5) * uGrain;

      gl_FragColor = color;
    }
  `,
};

/** The vertex-fetch uniforms every material on the mesh shares. */
interface SharedUniforms {
  uPositions: { value: THREE.Texture };
  uNormals: { value: THREE.Texture };
  uVertexSize: { value: THREE.Vector2 };
}

function createSurfaceMaterial(
  palette: Palette,
  shared: SharedUniforms,
): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    vertexShader: SURFACE_VERTEX_SHADER,
    fragmentShader: SURFACE_FRAGMENT_SHADER,
    uniforms: {
      ...shared,
      uDeep: { value: new THREE.Color(palette.deep) },
      uLit: { value: new THREE.Color(palette.lit) },
      uFill: { value: new THREE.Color(palette.fill) },
      uFillStrength: { value: palette.fillStrength },
      uRim: { value: new THREE.Color(palette.rim) },
      uRimPower: { value: palette.rimPower },
      uRimStrength: { value: palette.rimStrength },
      uSpecular: { value: palette.specular },
      uEmissive: { value: new THREE.Color(palette.emissive) },
      uEmissiveStrength: { value: palette.emissiveStrength },
    },
    // Push the surface back a hair so the wireframe drawn on top of it does
    // not z-fight along every edge.
    polygonOffset: true,
    polygonOffsetFactor: 1,
    polygonOffsetUnits: 1,
  });
}

export interface ViewerOptions {
  bloomStrength?: number;
  bloomRadius?: number;
  bloomThreshold?: number;
  wireColor?: number;
  wireOpacity?: number;
}

/** Owns the Three.js scene and draws a GNM mesh into it. */
export class Viewer {
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly controls: OrbitControls;

  private readonly renderer: THREE.WebGLRenderer;
  private readonly composer: EffectComposer;
  private readonly bloom: UnrealBloomPass;
  private readonly finish: ShaderPass;
  private readonly geometry = new THREE.BufferGeometry();
  private readonly wireGeometry = new THREE.BufferGeometry();
  private readonly wireMaterial: THREE.ShaderMaterial;
  private readonly wire: THREE.LineSegments;
  private readonly correspondence: THREE.Points | null;
  private readonly pivot = new THREE.Group();
  readonly gpu: GpuEvaluator;

  constructor(canvas: HTMLCanvasElement, model: GnmModel, options: ViewerOptions = {}) {
    const {
      bloomStrength = 0.38,
      bloomRadius = 0.45,
      // High enough that only the rim, the highlights and the wire cross it.
      // At a low threshold the broad surface blooms too and the head turns
      // into a featureless lamp.
      bloomThreshold = 0.82,
      wireColor = BRAND.tealPale,
      wireOpacity = 0.35,
    } = options;

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.1;
    this.renderer.setClearColor(BACKGROUND, 1);

    this.camera = new THREE.PerspectiveCamera(28, 1, 0.01, 100);

    this.gpu = new GpuEvaluator(this.renderer, model);
    const shared: SharedUniforms = {
      uPositions: { value: this.gpu.positionTexture },
      uNormals: { value: this.gpu.normalTexture },
      uVertexSize: { value: this.gpu.vertexSize },
    };

    const vertexCount = model.vertexCount;
    // The template stands in as `position` so Three.js can compute bounds; the
    // shader ignores it and fetches the posed value by index instead.
    const template = new THREE.BufferAttribute(model.template, 3);
    const indexAttribute = new THREE.BufferAttribute(
      Float32Array.from({ length: vertexCount }, (_, i) => i), 1,
    );

    const { indices, groups } = model.groupTrianglesByComponent();
    this.geometry.setAttribute('position', template);
    this.geometry.setAttribute('aIndex', indexAttribute);
    this.geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    const materials = groups.map((group, slot) => {
      this.geometry.addGroup(group.start, group.count, slot);
      return createSurfaceMaterial(COMPONENT_PALETTES[group.component], shared);
    });

    const mesh = new THREE.Mesh(this.geometry, materials);
    // The bounding sphere is computed from the template, but the posed mesh
    // can leave it, so culling would pop the head out of frame.
    mesh.frustumCulled = false;
    this.pivot.add(mesh);

    // The wire reads the very same textures as the surface, so it can never
    // drift out of sync with the shape and costs no extra upload.
    this.wireGeometry.setAttribute('position', template);
    this.wireGeometry.setAttribute('aIndex', indexAttribute);
    this.wireGeometry.setIndex(new THREE.BufferAttribute(model.quadEdges, 1));
    this.wireMaterial = new THREE.ShaderMaterial({
      vertexShader: SURFACE_VERTEX_SHADER,
      fragmentShader: WIREFRAME_FRAGMENT_SHADER,
      uniforms: {
        ...shared,
        uColor: { value: new THREE.Color(wireColor) },
        uOpacity: { value: wireOpacity },
      },
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    this.wire = new THREE.LineSegments(this.wireGeometry, this.wireMaterial);
    this.wire.frustumCulled = false;
    this.pivot.add(this.wire);

    this.correspondence = this.buildCorrespondencePoints(model, shared);
    if (this.correspondence) this.pivot.add(this.correspondence);

    this.scene.add(this.pivot);

    this.composer = new EffectComposer(this.renderer);
    // The clear colour is passed explicitly rather than inherited. RenderPass
    // clears the buffer *before* it renders the scene, so with no colour of
    // its own it clears with whatever GL last latched -- and the bloom pass
    // latches black. Any frame carrying an extra pass, which is exactly a
    // frame where the mesh was re-evaluated, then drew a black background.
    this.composer.addPass(
      new RenderPass(this.scene, this.camera, null, new THREE.Color(BACKGROUND), 1),
    );
    this.bloom = new UnrealBloomPass(
      new THREE.Vector2(1, 1), bloomStrength, bloomRadius, bloomThreshold,
    );
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());
    this.finish = new ShaderPass(FINISH_SHADER);
    this.composer.addPass(this.finish);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.enablePan = false;
  }

  private buildCorrespondencePoints(
    model: GnmModel,
    shared: SharedUniforms,
  ): THREE.Points | null {
    const mapping = model.correspondence;
    if (!mapping) return null;

    const count = mapping.vertices.length;
    const geometry = new THREE.BufferGeometry();
    const indices = new Float32Array(count);
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const vertex = mapping.vertices[i];
      indices[i] = vertex;
      // Template positions again, only so Three.js can size a bounding
      // sphere; the shader fetches the posed value.
      positions[i * 3] = model.template[vertex * 3];
      positions[i * 3 + 1] = model.template[vertex * 3 + 1];
      positions[i * 3 + 2] = model.template[vertex * 3 + 2];
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('aIndex', new THREE.BufferAttribute(indices, 1));
    geometry.setAttribute(
      'aColor', new THREE.BufferAttribute(mapping.colors, 3, true),
    );

    const points = new THREE.Points(
      geometry,
      new THREE.ShaderMaterial({
        vertexShader: CORRESPONDENCE_VERTEX_SHADER,
        fragmentShader: CORRESPONDENCE_FRAGMENT_SHADER,
        uniforms: {
          ...shared,
          uPointSize: { value: 3.4 },
        },
        transparent: true,
        // Tested against depth so points on the far side of the head stay
        // hidden, but not written, so overlapping markers still blend.
        depthWrite: false,
      }),
    );
    points.frustumCulled = false;
    points.visible = false;
    return points;
  }

  setCorrespondenceVisible(visible: boolean): void {
    if (this.correspondence) this.correspondence.visible = visible;
  }

  get hasCorrespondence(): boolean {
    return this.correspondence !== null;
  }

  /** Frames the camera on a mesh and centres the pivot on it. */
  frame(vertices: Float32Array): void {
    const box = new THREE.Box3();
    const point = new THREE.Vector3();
    for (let i = 0; i < vertices.length; i += 3) {
      box.expandByPoint(point.set(vertices[i], vertices[i + 1], vertices[i + 2]));
    }
    const centre = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());

    this.pivot.position.set(-centre.x, -centre.y, -centre.z);

    const extent = Math.max(size.x, size.y);
    const distance = extent / (2 * Math.tan((this.camera.fov * Math.PI) / 360)) * 1.6;
    this.camera.position.set(0, 0, distance);
    this.camera.near = distance / 100;
    this.camera.far = distance * 10;
    this.camera.updateProjectionMatrix();
    this.controls.target.set(0, 0, 0);
    this.controls.update();
  }

  /** Re-evaluates the mesh on the GPU for one set of parameters. */
  update(
    identity: ArrayLike<number>,
    expression: ArrayLike<number>,
    rotations: ArrayLike<number>,
    translation: ArrayLike<number>,
    correctives?: ArrayLike<number>,
  ): void {
    this.gpu.update(identity, expression, rotations, translation, correctives);
  }

  setWireframeVisible(visible: boolean): void {
    this.wire.visible = visible;
  }

  setWireframeOpacity(opacity: number): void {
    this.wireMaterial.uniforms.uOpacity.value = opacity;
  }

  setBloomStrength(strength: number): void {
    this.bloom.strength = strength;
  }

  setSize(width: number, height: number): void {
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
    this.composer.setSize(width, height);
    this.bloom.resolution.set(width, height);
  }

  render(elapsedSeconds: number): void {
    this.finish.uniforms.uTime.value = elapsedSeconds;
    this.controls.update();
    this.composer.render();
  }
}
