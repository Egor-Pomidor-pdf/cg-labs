#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;

// Набор 0: данные сцены, общие для всех фигур кадра.
layout(set = 0, binding = 0) uniform Scene {
	mat4 view;
	mat4 projection;
} scene;

// Набор 1: данные конкретной фигуры, свой у каждой.
layout(set = 1, binding = 0) uniform Object {
	mat4 model;
	vec4 tint;
} object;

layout(location = 0) out vec3 out_color;

void main() {
	gl_Position = scene.projection * scene.view * object.model * vec4(in_position, 1.0);
	out_color = in_color;
}
