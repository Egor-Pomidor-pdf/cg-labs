#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;

// Все внешние данные базовой части лежат в одном uniform-буфере.
layout(set = 0, binding = 0) uniform Uniforms {
	mat4 model;
	mat4 view_projection;
	mat4 normal_matrix;
	vec4 camera_position;
	vec4 light_direction; // направление НА источник
	vec4 light_color;     // rgb -- цвет, w -- интенсивность
	vec4 ambient;         // rgb -- цвет, w -- интенсивность
	vec4 albedo;
	vec4 specular;        // rgb -- цвет блика, w -- коэффициент блика
} u;

layout(location = 0) out vec3 out_position;
layout(location = 1) out vec3 out_normal;

void main() {
	// Освещение считается в мировом пространстве, поэтому во фрагментный шейдер
	// уходят мировая позиция и мировая нормаль.
	const vec4 world_position = u.model * vec4(in_position, 1.0);

	gl_Position = u.view_projection * world_position;
	out_position = world_position.xyz;
	out_normal = mat3(u.normal_matrix) * in_normal;
}
