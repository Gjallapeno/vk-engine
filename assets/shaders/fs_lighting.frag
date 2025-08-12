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

struct Ray { vec3 o; vec3 d; };

Ray makeRay(vec2 p) {
    vec2 ndc = (p / cam.renderResolution) * 2.0 - 1.0;
    vec4 h0 = cam.invViewProj * vec4(ndc, 0.0, 1.0);
    vec4 h1 = cam.invViewProj * vec4(ndc, 1.0, 1.0);
    vec3 ro = h0.xyz / h0.w;
    vec3 rd = normalize(h1.xyz / h1.w - ro);
    return Ray(ro, rd);
}

bool gridRaycastL0(Ray r, out int steps) {
    steps = 0;
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

    vec3 cellf = (pos - vox.min) / (vox.max - vox.min) * vec3(vox.dim);
    ivec3 cell = ivec3(clamp(floor(cellf), vec3(0.0), vec3(vox.dim) - vec3(1.0)));

    ivec3 step = ivec3(greaterThan(r.d, vec3(0.0))) * 2 - ivec3(1);
    vec3 cellSize = (vox.max - vox.min) / vec3(vox.dim);
    vec3 next = vox.min + (vec3(cell) + (vec3(step)+1.0)*0.5) * cellSize;
    vec3 tMax = (next - pos) / r.d;
    vec3 tDelta = cellSize / abs(r.d);
    for(int i=0;i<1024;i++){
        if(any(lessThan(cell, ivec3(0))) || any(greaterThanEqual(cell, vox.dim))) break;
        steps++;
        if(texelFetch(uOccTex, cell, 0).r > 0u) return true;
        if(tMax.x < tMax.y){
            if(tMax.x < tMax.z){ cell.x += step.x; t = tMax.x; tMax.x += tDelta.x; }
            else{ cell.z += step.z; t = tMax.z; tMax.z += tDelta.z; }
        }else{
            if(tMax.y < tMax.z){ cell.y += step.y; t = tMax.y; tMax.y += tDelta.y; }
            else{ cell.z += step.z; t = tMax.z; tMax.z += tDelta.z; }
        }
        pos = r.o + t * r.d;
    }
    return false;
}

bool gridRaycast(Ray r, out int stepsL1, out int stepsL0) {
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

    ivec3 dim1 = textureSize(uOccTexL1, 0);
    vec3 cellSize = (vox.max - vox.min) / vec3(dim1);
    vec3 rel = (pos - vox.min) / (vox.max - vox.min);
    ivec3 cell1 = ivec3(clamp(floor(rel * vec3(dim1)), vec3(0.0), vec3(dim1) - vec3(1.0)));
    ivec3 step = ivec3(greaterThan(r.d, vec3(0.0))) * 2 - ivec3(1);
    vec3 next = vox.min + (vec3(cell1) + (vec3(step)+1.0)*0.5) * cellSize;
    vec3 tMax = (next - pos) / r.d;
    vec3 tDelta = cellSize / abs(r.d);
    for(int i=0;i<1024;i++){
        if(any(lessThan(cell1, ivec3(0))) || any(greaterThanEqual(cell1, dim1))) break;
        if(texelFetch(uOccTexL1, cell1, 0).r > 0u){
            Ray r2; r2.o = pos; r2.d = r.d;
            bool hit = gridRaycastL0(r2, stepsL0);
            if(hit) return true; else return false;
        }
        stepsL1++;
        if(tMax.x < tMax.y){
            if(tMax.x < tMax.z){ cell1.x += step.x; t = tMax.x; tMax.x += tDelta.x; }
            else{ cell1.z += step.z; t = tMax.z; tMax.z += tDelta.z; }
        }else{
            if(tMax.y < tMax.z){ cell1.y += step.y; t = tMax.y; tMax.y += tDelta.y; }
            else{ cell1.z += step.z; t = tMax.z; tMax.z += tDelta.z; }
        }
        pos = r.o + t * r.d;
    }
    return false;
}

void main() {
    vec2 uv = gl_FragCoord.xy / cam.outputResolution;
    vec3 albedo = texture(gAlbedoRough, uv).rgb;
    if (cam.debugLevel > 0.5 || cam.debugSteps > 0.5) {
        outColor = vec4(albedo,1.0);
        return;
    }
    vec3 normal = texture(gNormal, uv).xyz;
    float depth = texture(gDepth, uv).r;
    vec3 lightDir = normalize(vec3(-0.5, -1.0, -0.3));
    float ndl = max(dot(normal, -lightDir), 0.0);
    float vis = 1.0;
    if(ndl > 0.0){
        vec2 rp = uv * cam.renderResolution;
        Ray viewRay = makeRay(rp);
        vec3 pos = viewRay.o + viewRay.d * depth;
        pos += normal * 0.001;
        Ray sh; sh.o = pos; sh.d = -lightDir;
        int s1; int s0;
        bool hit = gridRaycast(sh, s1, s0);
        int totalSteps = s0 + s1;
        ivec2 coord = ivec2(gl_FragCoord.xy) / STEPS_SCALE;
        imageAtomicAdd(stepsImg, coord, uint(totalSteps));
        if(hit) vis = 0.0;
    }
    vec3 color = albedo * ndl * vis;
    if (cam.debugNormals > 0.5) color = normal * 0.5 + 0.5;
    outColor = vec4(color, 1.0);
}
