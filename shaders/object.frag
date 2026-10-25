#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;

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

layout(location = 0) out vec4 out_color;

void main() {
	// После интерполяции длина нормали уже не единичная.
	vec3 n = normalize(in_normal);
	const vec3 v = normalize(u.camera_position.xyz - in_position);

	// Отсечение граней выключено, так что видна может быть и обратная сторона.
	// Для неё нормаль разворачивается к наблюдателю, иначе грань будет чёрной.
	if (dot(n, v) < 0.0) {
		n = -n;
	}

	const vec3 l = normalize(u.light_direction.xyz);

	// Модель Блинн-Фонга: рассеянная составляющая по закону Ламберта, зеркальная --
	// через вектор полупути между направлением на свет и на наблюдателя.
	const float diffuse = max(dot(n, l), 0.0);
	const vec3 h = normalize(l + v);
	const float specular = (diffuse > 0.0)
	                     ? pow(max(dot(n, h), 0.0), max(u.specular.w, 1.0))
	                     : 0.0;

	const vec3 light = u.light_color.rgb * u.light_color.w;

	const vec3 color = u.ambient.rgb * u.ambient.w * u.albedo.rgb +
	                   light * (u.albedo.rgb * diffuse + u.specular.rgb * specular);

	out_color = vec4(color, 1.0);
}
