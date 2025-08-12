#version 460
layout(location=0) out vec4 outColor;

layout(set=0, binding=0, std140) uniform Camera {
    mat4 invViewProj;
    vec2 resolution;
    float time;
    float _pad;
} cam;

layout(set=0, binding=1, std140) uniform VoxelAABB {
    vec3 min; float pad0;
    vec3 max; float pad1;
    ivec3 dim; int pad2;
} vox;

layout(set=0, binding=2) uniform usampler3D uOccTex;
layout(set=0, binding=3) uniform usampler3D uMatTex;

struct Ray { vec3 o; vec3 d; };

Ray makeRay(vec2 p) {
    vec2 ndc = (p / cam.resolution) * 2.0 - 1.0;
    vec4 h0 = cam.invViewProj * vec4(ndc, 0.0, 1.0);
    vec4 h1 = cam.invViewProj * vec4(ndc, 1.0, 1.0);
    vec3 ro = h0.xyz / h0.w;
    vec3 rd = normalize(h1.xyz / h1.w - ro);
    return Ray(ro, rd);
}

bool gridRaycast(Ray r, out ivec3 cell, out int hitFace) {
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
    cell = ivec3(clamp(floor(cellf), vec3(0.0), vec3(vox.dim) - vec3(1.0)));

    ivec3 step = ivec3(greaterThan(r.d, vec3(0.0))) * 2 - ivec3(1);
    vec3 cellSize = (vox.max - vox.min) / vec3(vox.dim);
    vec3 next = vox.min + (vec3(cell) + (vec3(step)+1.0)*0.5) * cellSize;
    vec3 tMax = (next - pos) / r.d;
    vec3 tDelta = cellSize / abs(r.d);
    hitFace = -1;
    for(int i=0;i<1024;i++){
        if(any(lessThan(cell, ivec3(0))) || any(greaterThanEqual(cell, vox.dim))) break;
        if(texelFetch(uOccTex, cell, 0).r > 0u) return true;
        if(tMax.x < tMax.y){
            if(tMax.x < tMax.z){
                cell.x += step.x; tMax.x += tDelta.x; hitFace = step.x > 0 ? 0 : 1;
            }else{
                cell.z += step.z; tMax.z += tDelta.z; hitFace = step.z > 0 ? 4 : 5;
            }
        }else{
            if(tMax.y < tMax.z){
                cell.y += step.y; tMax.y += tDelta.y; hitFace = step.y > 0 ? 2 : 3;
            }else{
                cell.z += step.z; tMax.z += tDelta.z; hitFace = step.z > 0 ? 4 : 5;
            }
        }
    }
    return false;
}

void main() {
    Ray r = makeRay(gl_FragCoord.xy);
    ivec3 cell; int face;
    vec3 col = vec3(0.0);
    if (gridRaycast(r, cell, face)) {
        uint m = texelFetch(uMatTex, cell, 0).r;
        if(m == 1u) col = vec3(0.55,0.27,0.07); // terrain
        else if(m == 2u) col = vec3(0.1,0.8,0.1); // tree
        else col = vec3(1.0);
    }
    outColor = vec4(col, 1.0);
}
