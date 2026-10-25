#version 450

layout(location = 0) in vec3 in_color;

layout(set = 1, binding = 0) uniform Object {
	mat4 model;
	vec4 tint;
} object;

layout(location = 0) out vec4 out_color;

void main() {
	// Процедурный цвет вершины умножается на цвет фигуры из её набора дескрипторов.
	out_color = vec4(in_color * object.tint.rgb, 1.0);
}
