#version 450

layout(set = 1, binding = 0) uniform Object {
	mat4 model;
	mat4 normal_matrix;
	vec4 albedo;
	vec4 specular;
} object;

layout(location = 0) out vec4 out_color;

void main() {
	// Цвет источника как есть, без освещения.
	out_color = vec4(object.albedo.rgb, 1.0);
}
