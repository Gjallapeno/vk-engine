#version 460
layout(location=0) out vec4 outColor;

layout(set=0, binding=0, std140) uniform Camera {
    mat4 invViewProj;
    vec2 renderResolution;
    vec2 outputResolution;
    float time;
    float debugNormals;
    float debugLevel;
    float debugSteps;
    vec4 pad0;
} cam;

layout(set=0, binding=1) uniform sampler2D gAlbedoRough;
layout(set=0, binding=2) uniform sampler2D gNormal;
layout(set=0, binding=3) uniform sampler2D gDepth;

layout(set=0, binding=4, std140) uniform VoxelAABB {
    vec3 min; float pad0;
    vec3 max; float pad1;
    ivec3 dim; int pad2;
} vox;

layout(set=0, binding=5) uniform usampler3D uOccTex;
layout(set=0, binding=6) uniform usampler3D uOccTexL1;
layout(set=0, binding=7, r32ui) uniform uimage2D stepsImg;

const int STEPS_SCALE = 4;
// Define sun direction as the direction light RAYS travel (sky → ground)
const vec3 sunDir = normalize(vec3(-0.5, -1.0, -0.3));

struct Ray { vec3 o; vec3 d; };

Ray makeRay(vec2 p) {
    vec2 ndc = (p / cam.renderResolution) * 2.0 - 1.0;
    vec4 h0 = cam.invViewProj * vec4(ndc, 0.0, 1.0);
    vec4 h1 = cam.invViewProj * vec4(ndc, 1.0, 1.0);
    vec3 ro = h0.xyz / h0.w;
    vec3 rd = normalize(h1.xyz / h1.w - ro);
    return Ray(ro, rd);
}

bool aabbHit(vec3 ro, vec3 rd, vec3 vmin, vec3 vmax, out float t0, out float t1) {
    vec3 invD = 1.0 / rd;
    vec3 t0s = (vmin - ro) * invD;
    vec3 t1s = (vmax - ro) * invD;
    vec3 tsm = min(t0s, t1s);
    vec3 tsM = max(t0s, t1s);
    t0 = max(max(tsm.x, tsm.y), tsm.z);
    t1 = min(min(tsM.x, tsM.y), tsM.z);
    return t1 > max(t0, 0.0);
}

// Robust L0 shadow test that advances into the next voxel before sampling
float shadowVisibilityL0_advanceFirst(vec3 P, vec3 L) {
    float t0, t1;
    if (!aabbHit(P, L, vox.min, vox.max, t0, t1)) return 1.0;
    t0 = max(t0, 0.0);

    const float EPS = 1e-4; // tiny epsilon to avoid boundary ambiguity
    vec3 start = P + L * max(t0, EPS) - vox.min;

    ivec3 step = ivec3(sign(L));
    ivec3 cell = ivec3(floor(start));
    vec3  ro   = start;
    vec3  next = vec3(cell) + max(step, ivec3(0));
    vec3  tMax   = (next - ro) / L;
    vec3  tDelta = abs(1.0 / L);

    // advance into the next cell before testing occupancy
    if (tMax.x < tMax.y) {
        if (tMax.x < tMax.z) { cell.x += step.x; tMax.x += tDelta.x; }
        else                  { cell.z += step.z; tMax.z += tDelta.z; }
    } else {
        if (tMax.y < tMax.z) { cell.y += step.y; tMax.y += tDelta.y; }
        else                  { cell.z += step.z; tMax.z += tDelta.z; }
    }

    const int MAX = 4096;
    for (int i=0; i<MAX; ++i) {
        if (any(lessThan(cell, ivec3(0))) || any(greaterThanEqual(cell, vox.dim))) return 1.0;
        if (texelFetch(uOccTex, cell, 0).r != 0u) return 0.0;

        if (tMax.x < tMax.y) {
            if (tMax.x < tMax.z) { cell.x += step.x; tMax.x += tDelta.x; }
            else                  { cell.z += step.z; tMax.z += tDelta.z; }
        } else {
            if (tMax.y < tMax.z) { cell.y += step.y; tMax.y += tDelta.y; }
            else                  { cell.z += step.z; tMax.z += tDelta.z; }
        }
    }
    return 1.0;
}

void main() {
    vec2 uv = gl_FragCoord.xy / cam.outputResolution;
    vec3 albedo = texture(gAlbedoRough, uv).rgb;
    if (cam.debugLevel > 0.5) {
        outColor = vec4(albedo, 1.0);
        return;
    }
    vec3 N = normalize(texture(gNormal, uv).xyz);
    float depth = texture(gDepth, uv).r;
    if (depth == 0.0) {
        outColor = vec4(albedo, 1.0);
        return;
    }
    vec2 rp = gl_FragCoord.xy / cam.outputResolution * cam.renderResolution;
    Ray viewRay = makeRay(rp);
    vec3 P = viewRay.o + viewRay.d * depth;

    // point→light direction
    vec3 L = normalize(-sunDir);

    // lambert first (independent of visibility)
    float ndl = max(dot(N, L), 0.0);

    // robust L0 visibility test
    float vis = shadowVisibilityL0_advanceFirst(P, L);

    vec3 color = albedo * ndl * vis;

    // Debug: show ndl and normal quickly
    if (cam.debugNormals > 0.5) color = N * 0.5 + 0.5;
    if (cam.debugSteps   > 0.5) color = vec3(ndl);

    outColor = vec4(color, 1.0);
}
