#version 450

layout(push_constant) uniform Push { float aspect; } PC;

void main() {
    const vec2 verts[3] = vec2[3](
        vec2(-0.6, -0.6),
        vec2( 0.0,  0.7),
        vec2( 0.6, -0.6)
    );

    vec2 p = verts[gl_VertexIndex];
    gl_Position = vec4(p.x, p.y / max(PC.aspect, 0.0001), 0.0, 1.0);
}
