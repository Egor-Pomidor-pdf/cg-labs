#version 450

// Шейдер для значков источников света и конусов прожекторов.
// Нормаль и процедурный цвет вершины здесь не нужны: фигуры рисуются
// постоянным цветом и не участвуют в расчёте освещения.

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 in_color;

layout(set = 0, binding = 0) uniform Scene {
	mat4 view_projection;
	vec4 camera_position;
	vec4 ambient;
	vec4 light_direction;
	vec4 light_color;
	uvec4 light_counts;
} scene;

layout(set = 1, binding = 0) uniform Object {
	mat4 model;
	mat4 normal_matrix;
	vec4 albedo;
	vec4 specular;
} object;

void main() {
	gl_Position = scene.view_projection * object.model * vec4(in_position, 1.0);
}
