#version 460
layout(location=0) out vec4 outColor;

layout(set=0, binding=0, std140) uniform Camera {
    mat4 invViewProj;
    vec2 resolution;
    float time;
    float debugNormals;
} cam;

layout(set=0, binding=1) uniform sampler2D gAlbedoRough;
layout(set=0, binding=2) uniform sampler2D gNormal;
layout(set=0, binding=3) uniform sampler2D gDepth;

void main() {
    ivec2 uv = ivec2(gl_FragCoord.xy);
    vec3 albedo = texelFetch(gAlbedoRough, uv, 0).rgb;
    vec3 normal = texelFetch(gNormal, uv, 0).xyz;
    float depth = texelFetch(gDepth, uv, 0).r;
    vec3 lightDir = normalize(vec3(-0.5, -1.0, -0.3));
    float ndl = max(dot(normal, -lightDir), 0.0);
    vec3 color = albedo * ndl;
    if (cam.debugNormals > 0.5) color = normal * 0.5 + 0.5;
    outColor = vec4(color, 1.0);
}
