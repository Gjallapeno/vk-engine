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
    vec3 color = albedo * ndl;
    if (cam.debugNormals > 0.5) color = normal * 0.5 + 0.5;
    outColor = vec4(color, 1.0);
}
