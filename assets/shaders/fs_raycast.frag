#version 460
layout(location=0) out vec4 outAlbedoRough;
layout(location=1) out vec4 outNormal;
layout(location=2) out float outDepth;

layout(set=0, binding=0, std140) uniform Camera {
    mat4 invViewProj;
    vec2 renderResolution;
    vec2 outputResolution;
    float time;
    float debugNormals; // reused in lighting pass
    float debugLevel;
    float debugSteps;
    vec4 pad0;
} cam;

layout(set=0, binding=1, std140) uniform VoxelAABB {
    vec3 min; float pad0;
    vec3 max; float pad1;
    ivec3 dim; int pad2;
    ivec3 occL1Dim; int pad3;
    vec3 occL1CellSize; float pad4;
} vox;

layout(set=0, binding=2) uniform usampler3D uOccTex;
layout(set=0, binding=3) uniform usampler3D uMatTex;
layout(set=0, binding=4) uniform usampler3D uOccTexL1;
layout(set=0, binding=5, r32ui) uniform uimage2D stepsImg;

const int STEPS_SCALE = 4;

const vec3 N[6] = vec3[6](
    vec3( 1,0,0), vec3(-1,0,0),
    vec3( 0,1,0), vec3( 0,-1,0),
    vec3( 0,0,1), vec3( 0,0,-1)
);

const vec3 ALBEDO[4] = vec3[4](
    vec3(1.0),
    vec3(0.55,0.27,0.07),
    vec3(0.1,0.8,0.1),
    vec3(0.5,0.5,0.5)
);

struct Ray { vec3 o; vec3 d; };

Ray makeRay(vec2 p) {
    vec2 ndc = (p / cam.renderResolution) * 2.0 - 1.0;
    vec4 h0 = cam.invViewProj * vec4(ndc, 0.0, 1.0);
    vec4 h1 = cam.invViewProj * vec4(ndc, 1.0, 1.0);
    vec3 ro = h0.xyz / h0.w;
    vec3 rd = normalize(h1.xyz / h1.w - ro);
    return Ray(ro, rd);
}

// Fine DDA on the full-resolution voxel grid
bool gridRaycastL0(Ray r, vec3 invD, out ivec3 cell, out int hitFace, out float tHit, out int steps) {
    steps = 0;
    vec3 extent = vox.max - vox.min;
    vec3 cellSize = extent / vec3(vox.dim);
    vec3 invCell = 1.0 / cellSize;
    vec3 t0s = (vox.min - r.o) * invD;
    vec3 t1s = (vox.max - r.o) * invD;
    vec3 tsm = min(t0s, t1s);
    vec3 tsM = max(t0s, t1s);
    float t0 = max(max(tsm.x, tsm.y), tsm.z);
    float t1 = min(min(tsM.x, tsM.y), tsM.z);
    if (t1 <= max(t0, 0.0)) return false;
    float t = max(t0, 0.0);
    vec3 pos = r.o + t * r.d;

    vec3 cellf = (pos - vox.min) * invCell;
    cell = ivec3(clamp(floor(cellf), vec3(0.0), vec3(vox.dim) - vec3(1.0)));

    ivec3 step = ivec3(greaterThan(r.d, vec3(0.0))) * 2 - ivec3(1);
    vec3 next = vox.min + (vec3(cell) + (vec3(step)+1.0)*0.5) * cellSize;
    vec3 tMax = (next - pos) * invD;
    vec3 tDelta = cellSize * abs(invD);
    hitFace = -1;
    int maxSteps = int(dot(vec3(vox.dim), vec3(1)));
    for(int i=0;i<maxSteps;i++){
        if(any(lessThan(cell, ivec3(0))) || any(greaterThanEqual(cell, vox.dim))) break;
        steps++;
        if(texelFetch(uOccTex, cell, 0).r > 0u){ tHit = t; return true; }
        int a = (tMax.x < tMax.y) ? 0 : 1;
        a = (tMax[a] < tMax.z) ? a : 2;
        cell[a] += step[a];
        t       = tMax[a];
        tMax[a] += tDelta[a];
        hitFace = (step[a] > 0) ? (a*2) : (a*2+1);
    }
    return false;
}

// Traverse coarse L1 occupancy first, then descend to L0 if needed
bool gridRaycast(Ray r, out ivec3 cell, out int hitFace, out float tHit, out int stepsL1, out int stepsL0) {
    stepsL1 = 0;
    stepsL0 = 0;
    vec3 invD = 1.0 / r.d;
    vec3 t0s = (vox.min - r.o) * invD;
    vec3 t1s = (vox.max - r.o) * invD;
    vec3 tsm = min(t0s, t1s);
    vec3 tsM = max(t0s, t1s);
    float t0 = max(max(tsm.x, tsm.y), tsm.z);
    float t1 = min(min(tsM.x, tsM.y), tsM.z);
    if (t1 <= max(t0, 0.0)) return false;
    float t = max(t0, 0.0);
    vec3 pos = r.o + t * r.d;

    ivec3 dim1 = vox.occL1Dim;
    vec3 cellSize = vox.occL1CellSize;
    vec3 invCell = 1.0 / cellSize;
    vec3 cellf = (pos - vox.min) * invCell;
    ivec3 cell1 = ivec3(clamp(floor(cellf), vec3(0.0), vec3(dim1) - vec3(1.0)));
    ivec3 step = ivec3(greaterThan(r.d, vec3(0.0))) * 2 - ivec3(1);
    vec3 next = vox.min + (vec3(cell1) + (vec3(step) + 1.0) * 0.5) * cellSize;
    vec3 tMax = (next - pos) * invD;
    vec3 tDelta = cellSize * abs(invD);
    int maxSteps = int(dot(vec3(dim1), vec3(1)));
    for(int i=0;i<maxSteps;i++){
        if(any(lessThan(cell1, ivec3(0))) || any(greaterThanEqual(cell1, dim1))) break;
        if(texelFetch(uOccTexL1, cell1, 0).r > 0u){
            Ray r2; r2.o = r.o + t * r.d; r2.d = r.d;
            float tLocal; int s0;
            bool hit = gridRaycastL0(r2, invD, cell, hitFace, tLocal, s0);
            stepsL0 += s0;
            if(hit){ tHit = t + tLocal; return true; } else return false;
        }
        stepsL1++;
        int a = (tMax.x < tMax.y) ? 0 : 1;
        a = (tMax[a] < tMax.z) ? a : 2;
        cell1[a] += step[a];
        t       = tMax[a];
        tMax[a] += tDelta[a];
    }
    return false;
}

void main() {
    Ray r = makeRay(gl_FragCoord.xy);
    ivec3 cell; int face; float t; int steps1; int steps0;
    vec3 albedo = vec3(0.0);
    vec3 normal = vec3(0.0);
    float depth = 0.0;
    if (gridRaycast(r, cell, face, t, steps1, steps0)) {
        uint m = texelFetch(uMatTex, cell, 0).r;
        albedo = ALBEDO[m <= 3u ? int(m) : 0];
        normal = (face >= 0) ? N[face] : vec3(0);
        depth = t;
    }
    int totalSteps = steps0 + steps1;
    ivec2 coord = ivec2(gl_FragCoord.xy) / STEPS_SCALE;
    imageAtomicAdd(stepsImg, coord, uint(totalSteps));
    vec3 color = albedo;
    if(cam.debugLevel > 0.5) color = (steps0 > 0) ? vec3(0,1,0) : vec3(1,0,0);
    if(cam.debugSteps > 0.5) color = vec3(float(totalSteps) * 0.02);
    outAlbedoRough = vec4(color, 0.0);
    outNormal = vec4(normal, 0.0);
    outDepth = depth;
}
