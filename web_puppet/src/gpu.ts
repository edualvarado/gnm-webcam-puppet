/**
 * The GNM forward pass on the GPU.
 *
 * Positions are trivially parallel and would sit happily in the render vertex
 * shader. Normals are the reason this is a pipeline instead: smooth normals
 * need a vertex's neighbours, and a vertex shader cannot see them. Evaluating
 * each neighbour's basis inline would multiply the work by the vertex valence
 * -- up to 16 here -- so instead the positions are computed once into a
 * texture and a second pass gathers neighbours from it:
 *
 *   pass A   coefficients -> position texture   (one texel per vertex)
 *   pass B   position texture -> normal texture (gathers via adjacency)
 *   pass C   the render, fetching both by vertex index
 *
 * Pass A does all 153 components for every vertex unconditionally. The CPU
 * path skips zero coefficients, which is what made slider-dragging cheap, but
 * that optimisation stops paying the moment tracking drives every coefficient
 * at once -- which is exactly the case this exists for.
 */

import * as THREE from 'three';

import type { GnmModel } from './gnm.ts';

/** Width of the vertex-indexed textures; height follows from the count. */
const VERTEX_TEXTURE_WIDTH = 512;
/** Width of the basis textures, which are far larger. */
const BASIS_TEXTURE_WIDTH = 2048;

/** Maps a flat index to the centre of its texel. Shared by every pass. */
const TEXEL_LOOKUP = /* glsl */ `
  vec2 texelUv(float index, vec2 size) {
    float column = mod(index, size.x);
    float row = floor(index / size.x);
    return (vec2(column, row) + 0.5) / size;
  }
`;

function textureSize(count: number, width: number): THREE.Vector2 {
  return new THREE.Vector2(width, Math.ceil(count / width));
}

/**
 * Packs interleaved xyz triples into the RGBA a float texture requires.
 *
 * Three-channel float textures are not universally usable as sampler sources,
 * so the alpha channel is padding. It costs a third more memory and removes a
 * whole class of format-support problems.
 */
function packRgba<T extends Float32Array | Uint16Array>(
  source: T,
  texelCount: number,
  make: (length: number) => T,
  alpha?: Float32Array,
): T {
  const out = make(texelCount * 4);
  const triples = Math.min(texelCount, Math.floor(source.length / 3));
  for (let i = 0; i < triples; i++) {
    out[i * 4] = source[i * 3];
    out[i * 4 + 1] = source[i * 3 + 1];
    out[i * 4 + 2] = source[i * 3 + 2];
  }
  if (alpha) {
    for (let i = 0; i < triples; i++) out[i * 4 + 3] = alpha[i] as never;
  }
  return out;
}

function dataTexture(
  data: Float32Array | Uint16Array,
  size: THREE.Vector2,
  type: THREE.TextureDataType,
): THREE.DataTexture {
  const texture = new THREE.DataTexture(
    data, size.x, size.y, THREE.RGBAFormat, type,
  );
  // Nearest everywhere: these textures are addressed by exact index, so any
  // filtering would blend one vertex's data into its neighbour's.
  texture.minFilter = THREE.NearestFilter;
  texture.magFilter = THREE.NearestFilter;
  texture.needsUpdate = true;
  return texture;
}

function renderTarget(size: THREE.Vector2): THREE.WebGLRenderTarget {
  return new THREE.WebGLRenderTarget(size.x, size.y, {
    format: THREE.RGBAFormat,
    // Full float rather than half: this is what the verification reads back,
    // and it should not be the thing that limits the comparison.
    type: THREE.FloatType,
    minFilter: THREE.NearestFilter,
    magFilter: THREE.NearestFilter,
    depthBuffer: false,
  });
}

export class GpuEvaluator {
  readonly vertexSize: THREE.Vector2;

  private readonly renderer: THREE.WebGLRenderer;
  private readonly model: GnmModel;
  private readonly scene = new THREE.Scene();
  private readonly camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  private readonly quad: THREE.Mesh;

  private readonly positionMaterial: THREE.ShaderMaterial;
  private readonly normalMaterial: THREE.ShaderMaterial;
  private readonly positionTarget: THREE.WebGLRenderTarget;
  private readonly normalTarget: THREE.WebGLRenderTarget;

  private readonly jointMatrices: THREE.Matrix4[];
  private readonly readback: Float32Array;

  constructor(renderer: THREE.WebGLRenderer, model: GnmModel) {
    this.renderer = renderer;
    this.model = model;

    const vertexCount = model.vertexCount;
    this.vertexSize = textureSize(vertexCount, VERTEX_TEXTURE_WIDTH);
    const texels = this.vertexSize.x * this.vertexSize.y;

    // --- static inputs ------------------------------------------------------
    // The template carries its vertex's incident-triangle count in alpha,
    // which saves a whole texture and a fetch in the normal pass.
    const counts = new Float32Array(vertexCount);
    for (let i = 0; i < vertexCount; i++) counts[i] = model.normalAdjacencyCount[i];
    const templateTexture = dataTexture(
      packRgba(model.template, texels, (n) => new Float32Array(n), counts),
      this.vertexSize,
      THREE.FloatType,
    );

    const skinning = new Float32Array(texels * 4);
    for (let j = 0; j < model.jointCount; j++) {
      for (let v = 0; v < vertexCount; v++) {
        skinning[v * 4 + j] = model.skinningWeights[j * vertexCount + v];
      }
    }
    const skinningTexture = dataTexture(skinning, this.vertexSize, THREE.FloatType);

    // The bases go up as raw float16 bits -- exactly what the export wrote and
    // exactly what a half-float texture wants, so there is no decode step.
    const identityBits = new Uint16Array(
      model.buffer,
      model.manifest.arrays.identity_basis.byteOffset,
      model.manifest.arrays.identity_basis.byteLength / 2,
    );
    const expressionBits = new Uint16Array(
      model.buffer,
      model.manifest.arrays.expression_basis.byteOffset,
      model.manifest.arrays.expression_basis.byteLength / 2,
    );
    const identitySize = textureSize(
      model.identityDim * vertexCount, BASIS_TEXTURE_WIDTH,
    );
    const expressionSize = textureSize(
      model.expressionDim * vertexCount, BASIS_TEXTURE_WIDTH,
    );
    const identityTexture = dataTexture(
      packRgba(identityBits, identitySize.x * identitySize.y, (n) => new Uint16Array(n)),
      identitySize,
      THREE.HalfFloatType,
    );
    const expressionTexture = dataTexture(
      packRgba(expressionBits, expressionSize.x * expressionSize.y, (n) => new Uint16Array(n)),
      expressionSize,
      THREE.HalfFloatType,
    );

    // Correctives ride the same path as the bases. GLSL rejects a zero-length
    // uniform array and a zero-trip loop, so an export without them still
    // allocates one slot; its weight stays zero and the shape is never added.
    const correctiveSlots = Math.max(model.correctiveDim, 1);
    const correctiveBits = model.correctiveDim
      ? new Uint16Array(
          model.buffer,
          model.manifest.arrays.corrective_basis.byteOffset,
          model.manifest.arrays.corrective_basis.byteLength / 2,
        )
      : new Uint16Array(vertexCount * 3);
    const correctiveSize = textureSize(
      correctiveSlots * vertexCount, BASIS_TEXTURE_WIDTH,
    );
    const correctiveTexture = dataTexture(
      packRgba(correctiveBits, correctiveSize.x * correctiveSize.y, (n) => new Uint16Array(n)),
      correctiveSize,
      THREE.HalfFloatType,
    );

    const corners = model.maxNormalCorners;
    const adjacencySize = textureSize(vertexCount * corners, BASIS_TEXTURE_WIDTH);
    const adjacency = new Float32Array(adjacencySize.x * adjacencySize.y * 4);
    for (let i = 0; i < vertexCount * corners; i++) {
      adjacency[i * 4] = model.normalAdjacency[i * 2];
      adjacency[i * 4 + 1] = model.normalAdjacency[i * 2 + 1];
    }
    const adjacencyTexture = dataTexture(adjacency, adjacencySize, THREE.FloatType);

    // --- passes -------------------------------------------------------------
    this.positionTarget = renderTarget(this.vertexSize);
    this.normalTarget = renderTarget(this.vertexSize);

    this.jointMatrices = Array.from(
      { length: model.jointCount }, () => new THREE.Matrix4(),
    );

    this.positionMaterial = new THREE.ShaderMaterial({
      defines: {
        IDENTITY_COUNT: model.identityDim,
        EXPRESSION_COUNT: model.expressionDim,
        CORRECTIVE_COUNT: correctiveSlots,
        JOINT_COUNT: model.jointCount,
      },
      uniforms: {
        uTemplate: { value: templateTexture },
        uIdentityBasis: { value: identityTexture },
        uExpressionBasis: { value: expressionTexture },
        uCorrectiveBasis: { value: correctiveTexture },
        uSkinning: { value: skinningTexture },
        uIdentity: { value: new Float32Array(model.identityDim) },
        uExpression: { value: new Float32Array(model.expressionDim) },
        uCorrective: { value: new Float32Array(correctiveSlots) },
        uJoints: { value: this.jointMatrices },
        uVertexSize: { value: this.vertexSize },
        uIdentitySize: { value: identitySize },
        uExpressionSize: { value: expressionSize },
        uCorrectiveSize: { value: correctiveSize },
        uVertexCount: { value: vertexCount },
      },
      vertexShader: /* glsl */ `
        void main() { gl_Position = vec4(position.xy, 0.0, 1.0); }
      `,
      fragmentShader: /* glsl */ `
        uniform sampler2D uTemplate;
        uniform sampler2D uIdentityBasis;
        uniform sampler2D uExpressionBasis;
        uniform sampler2D uCorrectiveBasis;
        uniform sampler2D uSkinning;
        uniform float uIdentity[IDENTITY_COUNT];
        uniform float uExpression[EXPRESSION_COUNT];
        uniform float uCorrective[CORRECTIVE_COUNT];
        uniform mat4 uJoints[JOINT_COUNT];
        uniform vec2 uVertexSize;
        uniform vec2 uIdentitySize;
        uniform vec2 uExpressionSize;
        uniform vec2 uCorrectiveSize;
        uniform float uVertexCount;
        ${TEXEL_LOOKUP}

        void main() {
          vec2 texel = floor(gl_FragCoord.xy);
          float index = texel.y * uVertexSize.x + texel.x;

          vec2 uv = (texel + 0.5) / uVertexSize;
          vec3 vertex = texture2D(uTemplate, uv).xyz;

          for (int i = 0; i < IDENTITY_COUNT; i++) {
            vertex += uIdentity[i] * texture2D(
              uIdentityBasis,
              texelUv(float(i) * uVertexCount + index, uIdentitySize)
            ).xyz;
          }
          for (int i = 0; i < EXPRESSION_COUNT; i++) {
            vertex += uExpression[i] * texture2D(
              uExpressionBasis,
              texelUv(float(i) * uVertexCount + index, uExpressionSize)
            ).xyz;
          }
          for (int i = 0; i < CORRECTIVE_COUNT; i++) {
            vertex += uCorrective[i] * texture2D(
              uCorrectiveBasis,
              texelUv(float(i) * uVertexCount + index, uCorrectiveSize)
            ).xyz;
          }

          vec4 weights = texture2D(uSkinning, uv);
          vec4 homogeneous = vec4(vertex, 1.0);
          vec3 posed = vec3(0.0);
          for (int j = 0; j < JOINT_COUNT; j++) {
            posed += weights[j] * (uJoints[j] * homogeneous).xyz;
          }
          gl_FragColor = vec4(posed, 1.0);
        }
      `,
    });

    this.normalMaterial = new THREE.ShaderMaterial({
      defines: { MAX_CORNERS: corners },
      uniforms: {
        uPositions: { value: this.positionTarget.texture },
        uAdjacency: { value: adjacencyTexture },
        uTemplate: { value: templateTexture },
        uVertexSize: { value: this.vertexSize },
        uAdjacencySize: { value: adjacencySize },
      },
      vertexShader: /* glsl */ `
        void main() { gl_Position = vec4(position.xy, 0.0, 1.0); }
      `,
      fragmentShader: /* glsl */ `
        uniform sampler2D uPositions;
        uniform sampler2D uAdjacency;
        uniform sampler2D uTemplate;
        uniform vec2 uVertexSize;
        uniform vec2 uAdjacencySize;
        ${TEXEL_LOOKUP}

        void main() {
          vec2 texel = floor(gl_FragCoord.xy);
          float index = texel.y * uVertexSize.x + texel.x;
          vec2 uv = (texel + 0.5) / uVertexSize;

          vec3 origin = texture2D(uPositions, uv).xyz;
          float corners = texture2D(uTemplate, uv).w;

          // The cross product of two edge vectors taken from any corner of a
          // triangle in cyclic order is the same vector, so this reproduces
          // the CPU's per-face normal exactly -- magnitude included, which is
          // twice the area and therefore the weighting.
          vec3 accumulated = vec3(0.0);
          for (int k = 0; k < MAX_CORNERS; k++) {
            if (float(k) >= corners) break;
            vec2 pair = texture2D(
              uAdjacency,
              texelUv(index * float(MAX_CORNERS) + float(k), uAdjacencySize)
            ).xy;
            vec3 first = texture2D(uPositions, texelUv(pair.x, uVertexSize)).xyz;
            vec3 second = texture2D(uPositions, texelUv(pair.y, uVertexSize)).xyz;
            accumulated += cross(first - origin, second - origin);
          }

          float length = max(length(accumulated), 1e-12);
          gl_FragColor = vec4(accumulated / length, 1.0);
        }
      `,
    });

    this.quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.positionMaterial);
    this.quad.frustumCulled = false;
    this.scene.add(this.quad);

    this.readback = new Float32Array(texels * 4);
  }

  get positionTexture(): THREE.Texture {
    return this.positionTarget.texture;
  }

  get normalTexture(): THREE.Texture {
    return this.normalTarget.texture;
  }

  /** Runs both passes for one set of parameters. */
  update(
    identity: ArrayLike<number>,
    expression: ArrayLike<number>,
    rotations: ArrayLike<number>,
    translation: ArrayLike<number>,
    correctives?: ArrayLike<number>,
  ): void {
    const transforms = this.model.jointTransforms(identity, rotations, translation);
    for (let j = 0; j < this.model.jointCount; j++) {
      const o = j * 12;
      // Three.js Matrix4.set takes row-major arguments, and the bottom row is
      // the implicit affine one.
      this.jointMatrices[j].set(
        transforms[o], transforms[o + 1], transforms[o + 2], transforms[o + 3],
        transforms[o + 4], transforms[o + 5], transforms[o + 6], transforms[o + 7],
        transforms[o + 8], transforms[o + 9], transforms[o + 10], transforms[o + 11],
        0, 0, 0, 1,
      );
    }

    (this.positionMaterial.uniforms.uIdentity.value as Float32Array).set(identity as ArrayLike<number> as never);
    (this.positionMaterial.uniforms.uExpression.value as Float32Array).set(expression as ArrayLike<number> as never);
    const correctiveWeights = this.positionMaterial.uniforms.uCorrective.value as Float32Array;
    correctiveWeights.fill(0);
    if (correctives) correctiveWeights.set(correctives as ArrayLike<number> as never, 0);

    const previousTarget = this.renderer.getRenderTarget();
    // These two passes borrow the viewer's renderer, so they have to hand it
    // back as they found it. Clearing is off because the fragment shader
    // writes every texel of both targets anyway -- the clear was pure
    // bandwidth, and it was also what put a stray clear colour into the shared
    // GL state partway through a frame.
    const previousAutoClear = this.renderer.autoClear;
    this.renderer.autoClear = false;

    this.quad.material = this.positionMaterial;
    this.renderer.setRenderTarget(this.positionTarget);
    this.renderer.render(this.scene, this.camera);

    this.quad.material = this.normalMaterial;
    this.renderer.setRenderTarget(this.normalTarget);
    this.renderer.render(this.scene, this.camera);

    this.renderer.setRenderTarget(previousTarget);
    this.renderer.autoClear = previousAutoClear;
  }

  /**
   * Reads posed positions back off the GPU, for comparison against the CPU.
   *
   * Synchronous and slow by design -- this is a correctness check, not
   * something the frame loop should call.
   *
   * @returns Interleaved xyz positions, `vertexCount * 3`.
   */
  readPositions(): Float32Array {
    this.renderer.readRenderTargetPixels(
      this.positionTarget, 0, 0, this.vertexSize.x, this.vertexSize.y, this.readback,
    );
    const out = new Float32Array(this.model.vertexCount * 3);
    for (let v = 0; v < this.model.vertexCount; v++) {
      out[v * 3] = this.readback[v * 4];
      out[v * 3 + 1] = this.readback[v * 4 + 1];
      out[v * 3 + 2] = this.readback[v * 4 + 2];
    }
    return out;
  }
}
