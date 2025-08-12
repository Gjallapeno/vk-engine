#version 450
layout(location = 0) out vec4 outColor;
void main() {
    outColor = vec4(gl_FragCoord.x / 1280.0, gl_FragCoord.y / 720.0, 0.3, 1.0);
}
