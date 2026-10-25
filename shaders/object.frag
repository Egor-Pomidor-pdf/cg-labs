#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 in_color;

layout(set = 0, binding = 0) uniform Scene {
	mat4 view_projection;
	vec4 camera_position;
	vec4 ambient;         // rgb -- цвет, w -- интенсивность
	vec4 light_direction; // направление НА направленный источник
	vec4 light_color;     // rgb -- цвет, w -- интенсивность
	uvec4 light_counts;   // x -- точечных источников, y -- прожекторных
} scene;

struct PointLight {
	vec4 position;
	vec4 color; // rgb -- цвет, w -- интенсивность
};

struct SpotLight {
	vec4 position;
	vec4 direction; // куда светит
	vec4 color;     // rgb -- цвет, w -- интенсивность
	vec4 cutoff;    // x -- косинус внутреннего угла, y -- внешнего
};

// Массивы источников объявлены без размера: сколько их на самом деле,
// шейдер узнаёт из scene.light_counts, а не из длины массива.
layout(std430, set = 0, binding = 1) readonly buffer PointLights {
	PointLight lights[];
} point_lights;

layout(std430, set = 0, binding = 2) readonly buffer SpotLights {
	SpotLight lights[];
} spot_lights;

layout(set = 1, binding = 0) uniform Object {
	mat4 model;
	mat4 normal_matrix;
	vec4 albedo;   // rgb -- цвет, w -- доля процедурного цвета вершины
	vec4 specular; // rgb -- цвет блика, w -- коэффициент блика
} object;

layout(location = 0) out vec4 out_color;

// Одна порция света по модели Блинн-Фонга: рассеянная составляющая по закону
// Ламберта и зеркальная -- через вектор полупути между направлением на
// источник и направлением на наблюдателя.
vec3 blinnPhong(vec3 n, vec3 v, vec3 l, vec3 radiance, vec3 albedo) {
	const float diffuse = max(dot(n, l), 0.0);
	if (diffuse <= 0.0) {
		return vec3(0.0); // поверхность отвёрнута от источника
	}

	const vec3 h = normalize(l + v);
	const float specular = pow(max(dot(n, h), 0.0), max(object.specular.w, 1.0));

	return radiance * (albedo * diffuse + object.specular.rgb * specular);
}

void main() {
	// После интерполяции длина нормали уже не единичная.
	vec3 n = normalize(in_normal);
	const vec3 v = normalize(scene.camera_position.xyz - in_position);

	// Отсечение граней выключено, так что видна может быть и обратная сторона.
	// Для неё нормаль разворачивается к наблюдателю, иначе грань будет чёрной.
	if (dot(n, v) < 0.0) {
		n = -n;
	}

	const vec3 albedo = object.albedo.rgb * mix(vec3(1.0), in_color, object.albedo.w);

	vec3 color = scene.ambient.rgb * scene.ambient.w * albedo;

	// Направленный источник: он бесконечно далеко, затухания нет.
	color += blinnPhong(n, v, normalize(scene.light_direction.xyz),
	                    scene.light_color.rgb * scene.light_color.w, albedo);

	// Точечные источники: светят во все стороны, яркость падает как 1/d^2.
	for (uint i = 0u; i < scene.light_counts.x; ++i) {
		const vec3 to_light = point_lights.lights[i].position.xyz - in_position;
		const float distance = length(to_light);
		if (distance <= 0.0) {
			continue;
		}

		const vec3 radiance = point_lights.lights[i].color.rgb *
		                      point_lights.lights[i].color.w / (distance * distance);

		color += blinnPhong(n, v, to_light / distance, radiance, albedo);
	}

	// Прожекторы: то же затухание плюс ограничение по углу. Между внутренним и
	// внешним углом яркость плавно падает до нуля -- это и есть мягкий край.
	for (uint i = 0u; i < scene.light_counts.y; ++i) {
		const vec3 to_light = spot_lights.lights[i].position.xyz - in_position;
		const float distance = length(to_light);
		if (distance <= 0.0) {
			continue;
		}

		const vec3 l = to_light / distance;

		// Косинус угла между осью прожектора и направлением на фрагмент.
		const float cosine = dot(-l, normalize(spot_lights.lights[i].direction.xyz));
		const float inner = spot_lights.lights[i].cutoff.x;
		const float outer = spot_lights.lights[i].cutoff.y;
		const float edge = clamp((cosine - outer) / max(inner - outer, 1e-4), 0.0, 1.0);
		if (edge <= 0.0) {
			continue; // фрагмент вне освещаемого конуса
		}

		const vec3 radiance = spot_lights.lights[i].color.rgb *
		                      spot_lights.lights[i].color.w * edge / (distance * distance);

		color += blinnPhong(n, v, l, radiance, albedo);
	}

	out_color = vec4(color, 1.0);
}
