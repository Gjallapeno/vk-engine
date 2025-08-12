#version 450

layout(push_constant) uniform Push { mat4 viewProj; } PC;

layout(location = 0) in vec3 inPos;   // x, y, z
layout(location = 1) in vec2 inUV;    // u, v

layout(location = 0) out vec2 vUV;

void main()
{
    vUV = inUV;
    gl_Position = PC.viewProj * vec4(inPos, 1.0);
}
