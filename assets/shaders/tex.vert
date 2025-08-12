#version 450

layout(push_constant) uniform Push { float aspect; } PC;

layout(location = 0) in vec2 inPos;   // x, y
layout(location = 1) in vec2 inUV;    // u, v

layout(location = 0) out vec2 vUV;

void main()
{
    vUV = inUV;
    // keep triangle/quad proportional across window sizes
    gl_Position = vec4(inPos.x, inPos.y / max(PC.aspect, 0.0001), 0.0, 1.0);
}
