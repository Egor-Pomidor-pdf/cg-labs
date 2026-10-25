#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 in_color;

// Набор 0: данные сцены, общие для всех фигур кадра.
layout(set = 0, binding = 0) uniform Scene {
	mat4 view_projection;
	vec4 camera_position;
	vec4 ambient;         // rgb -- цвет, w -- интенсивность
	vec4 light_direction; // направление НА направленный источник
	vec4 light_color;     // rgb -- цвет, w -- интенсивность
	uvec4 light_counts;   // x -- точечных источников, y -- прожекторных
} scene;

// Набор 1: данные конкретной фигуры и её материал.
layout(set = 1, binding = 0) uniform Object {
	mat4 model;
	mat4 normal_matrix;
	vec4 albedo;   // rgb -- цвет, w -- доля процедурного цвета вершины
	vec4 specular; // rgb -- цвет блика, w -- коэффициент блика
} object;

layout(location = 0) out vec3 out_position;
layout(location = 1) out vec3 out_normal;
layout(location = 2) out vec3 out_color;

void main() {
	// Освещение считается в мировом пространстве, поэтому дальше уходят
	// мировая позиция и мировая нормаль.
	const vec4 world_position = object.model * vec4(in_position, 1.0);

	gl_Position = scene.view_projection * world_position;
	out_position = world_position.xyz;
	out_normal = mat3(object.normal_matrix) * in_normal;
	out_color = in_color;
}
