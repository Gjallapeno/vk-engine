#version 450

layout(location = 0) in vec2 inPos;
layout(location = 1) in vec3 inCol;

layout(location = 0) out vec3 vCol;

layout(push_constant) uniform Push { float aspect; } PC;

void main() {
    vCol = inCol;
    // account for aspect so triangle stays tall when resizing
    gl_Position = vec4(inPos.x, inPos.y / max(PC.aspect, 0.0001), 0.0, 1.0);
}
