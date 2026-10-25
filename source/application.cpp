// Лабораторная работа №2 -- Камера и освещение
// Вариант 11: правильный икосаэдр
//
// Сцена из первой лабораторной (несколько икосаэдров на общей траектории)
// освещается по модели Блинн-Фонга: один направленный источник, массив
// точечных и массив прожекторных. Массивы лежат в storage-буферах, поэтому их
// длина не зашита в шейдер. Камера, источники и материалы настраиваются из
// интерфейса, источники можно показать значками и конусами.

#include "application.hpp"

#include <cstdint>
#include <cstddef>
#include <iostream>

#include <imgui.h>

#include "graphics.hpp"
#include "linalg.hpp"

namespace application {

namespace {

using linalg::Mat4;
using linalg::Vec3;

// Кроме позиции у вершины задана нормаль -- без неё освещение не посчитать.
// Процедурный цвет остался из первой лабораторной, он домножается на альбедо.
struct Vertex {
	Vec3 position;
	Vec3 normal;
	Vec3 color;
};

// Данные сцены: набор дескрипторов 0, привязывается один раз за кадр.
// Поля выровнены по 16 байт, как требует раскладка std140.
struct SceneUniforms {
	Mat4 view_projection;
	float camera_position[4];
	float ambient[4];         // rgb -- цвет, w -- интенсивность
	float light_direction[4]; // направление НА направленный источник
	float light_color[4];     // rgb -- цвет, w -- интенсивность
	uint32_t light_counts[4]; // x -- точечных источников, y -- прожекторных
};

// Данные одной фигуры и её материал: набор дескрипторов 1, свой у каждой.
struct ObjectUniforms {
	Mat4 model;
	Mat4 normal_matrix;
	float albedo[4];   // rgb -- цвет, w -- доля процедурного цвета вершины
	float specular[4]; // rgb -- цвет блика, w -- коэффициент блика
};

// Элементы storage-буферов -- ровно то же, что объявлено в шейдере.
struct PointLightData {
	float position[4];
	float color[4];
};

struct SpotLightData {
	float position[4];
	float direction[4];
	float color[4];
	float cutoff[4]; // x -- косинус внутреннего угла, y -- внешнего
};

constexpr uint32_t max_objects = 4;
constexpr uint32_t max_point_lights = 8;
constexpr uint32_t max_spot_lights = 4;

// Свой набор дескрипторов нужен каждой нарисованной фигуре: самим объектам,
// полу, значкам источников и конусам прожекторов.
constexpr uint32_t max_object_sets = max_objects + 1 + max_point_lights + max_spot_lights * 2;

constexpr uint32_t floor_set_index = max_objects;
constexpr uint32_t point_marker_set_index = floor_set_index + 1;
constexpr uint32_t spot_marker_set_index = point_marker_set_index + max_point_lights;
constexpr uint32_t cone_set_index = spot_marker_set_index + max_spot_lights;

// Параметры фигуры, которыми управляет интерфейс.
struct ObjectParams {
	float albedo[3];
	float scale; // множитель к общему растяжению
	float phase; // сдвиг по траектории, радианы
};

// Точечный источник: светит во все стороны из своей точки.
struct PointLight {
	float position[3];
	float color[3];
	float intensity;
};

// Прожектор: то же плюс направление и два угла -- внутренний и внешний.
struct SpotLight {
	float position[3];
	float direction[3];
	float color[3];
	float intensity;
	float inner_angle; // градусы, полный угол яркого пятна
	float outer_angle; // градусы, за ним света нет
};

constexpr uint32_t icosahedron_corner_count = 12; // уникальных вершин тела
constexpr uint32_t icosahedron_face_count = 20;   // треугольных граней
constexpr uint32_t icosahedron_vertex_count = icosahedron_face_count * 3;

constexpr uint32_t floor_vertex_count = 6; // два треугольника

// Конус прожектора: несколько лучей от вершины к ободу и сам обод.
constexpr uint32_t cone_ray_count = 8;
constexpr uint32_t cone_segment_count = 24;
constexpr uint32_t cone_vertex_count = (cone_ray_count + cone_segment_count) * 2;

constexpr uint32_t icosahedron_first_vertex = 0;
constexpr uint32_t floor_first_vertex = icosahedron_vertex_count;
constexpr uint32_t cone_first_vertex = floor_first_vertex + floor_vertex_count;
constexpr uint32_t total_vertex_count = cone_first_vertex + cone_vertex_count;

// 12 вершин правильного икосаэдра -- три взаимно перпендикулярных
// "золотых" прямоугольника со сторонами 2 и 2*phi.
// Длина каждого ребра равна 2, радиус описанной сферы -- sqrt(1 + phi^2).
constexpr float phi = 1.618033988749895f; // (1 + sqrt(5)) / 2

constexpr Vec3 icosahedron_corners[icosahedron_corner_count] = {
	{-1.0f,  phi,  0.0f},
	{ 1.0f,  phi,  0.0f},
	{-1.0f, -phi,  0.0f},
	{ 1.0f, -phi,  0.0f},
	{ 0.0f, -1.0f,  phi},
	{ 0.0f,  1.0f,  phi},
	{ 0.0f, -1.0f, -phi},
	{ 0.0f,  1.0f, -phi},
	{  phi,  0.0f, -1.0f},
	{  phi,  0.0f,  1.0f},
	{ -phi,  0.0f, -1.0f},
	{ -phi,  0.0f,  1.0f},
};

// 20 граней: тройки номеров вершин. Порядок вершин в каждой грани --
// против часовой стрелки, если смотреть на грань снаружи фигуры.
constexpr uint16_t icosahedron_faces[icosahedron_vertex_count] = {
	// 5 граней вокруг вершины 0
	 0, 11,  5,
	 0,  5,  1,
	 0,  1,  7,
	 0,  7, 10,
	 0, 10, 11,
	// 5 смежных с ними граней
	 1,  5,  9,
	 5, 11,  4,
	11, 10,  2,
	10,  7,  6,
	 7,  1,  8,
	// 5 граней вокруг вершины 3 (противоположной вершине 0)
	 3,  9,  4,
	 3,  4,  2,
	 3,  2,  6,
	 3,  6,  8,
	 3,  8,  9,
	// оставшиеся 5 граней
	 4,  9,  5,
	 2,  4, 11,
	 6,  2, 10,
	 8,  6,  7,
	 9,  8,  1,
};

graphics::Buffer vertex_buffer;

graphics::Buffer scene_uniform_buffer;
graphics::Buffer point_light_buffer;
graphics::Buffer spot_light_buffer;
graphics::Buffer object_uniform_buffers[max_object_sets];

VkDescriptorSetLayout vk_scene_set_layout = VK_NULL_HANDLE;
VkDescriptorSetLayout vk_object_set_layout = VK_NULL_HANDLE;
VkDescriptorPool vk_descriptor_pool = VK_NULL_HANDLE;
VkDescriptorSet vk_scene_set = VK_NULL_HANDLE;
VkDescriptorSet vk_object_sets[max_object_sets] = {};

VkPipelineLayout vk_pipeline_layout = VK_NULL_HANDLE;
VkPipeline vk_object_pipeline = VK_NULL_HANDLE; // освещённые фигуры
VkPipeline vk_light_pipeline = VK_NULL_HANDLE;  // значки источников
VkPipeline vk_line_pipeline = VK_NULL_HANDLE;   // конусы прожекторов

// Камера: позиция в мире и два угла поворота. При нулевых углах она смотрит
// вдоль -Z, то есть из точки {0, 1, 6} -- примерно в центр сцены.
Vec3 camera_position = {0.0f, 1.8f, 7.5f};
float camera_yaw = 0.0f;   // поворот вокруг вертикальной оси, градусы
float camera_pitch = -10.0f; // наклон вверх-вниз, градусы
float fov_degrees = 60.0f;

bool use_perspective = true;
float ortho_height = 3.5f; // половина высоты области видимости

// Направленный источник светит с бесконечности, поэтому задан только
// направлением. В интерфейсе задаётся направление НА источник.
Vec3 light_direction = {-0.4f, 0.8f, 0.5f};
float light_color[3] = {1.0f, 0.95f, 0.85f};
float light_intensity = 0.35f;

// Рассеянный свет -- приближение освещения, пришедшего со всех сторон.
float ambient_color[3] = {0.45f, 0.55f, 0.8f};
float ambient_intensity = 0.08f;

int point_light_count = 2;
PointLight point_lights[max_point_lights] = {
	{{ 2.2f, 1.6f,  2.0f}, {1.0f, 0.6f, 0.25f}, 12.0f},
	{{-2.4f, 1.2f, -1.6f}, {0.3f, 0.6f, 1.0f},  10.0f},
	{{ 0.0f, 2.0f,  0.0f}, {1.0f, 1.0f, 1.0f},   8.0f},
	{{ 0.0f, 2.0f,  0.0f}, {1.0f, 1.0f, 1.0f},   8.0f},
	{{ 0.0f, 2.0f,  0.0f}, {1.0f, 1.0f, 1.0f},   8.0f},
	{{ 0.0f, 2.0f,  0.0f}, {1.0f, 1.0f, 1.0f},   8.0f},
	{{ 0.0f, 2.0f,  0.0f}, {1.0f, 1.0f, 1.0f},   8.0f},
	{{ 0.0f, 2.0f,  0.0f}, {1.0f, 1.0f, 1.0f},   8.0f},
};

int spot_light_count = 1;
SpotLight spot_lights[max_spot_lights] = {
	{{0.0f, 4.0f, 0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f, 0.95f}, 45.0f, 14.0f, 26.0f},
	{{3.0f, 3.0f, 0.0f}, {-0.6f, -1.0f, 0.0f}, {0.6f, 1.0f, 0.7f}, 40.0f, 12.0f, 22.0f},
	{{0.0f, 3.0f, 0.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, 40.0f, 12.0f, 22.0f},
	{{0.0f, 3.0f, 0.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, 40.0f, 12.0f, 22.0f},
};

bool animate_lights = false;
float light_orbit_speed = 40.0f; // градусов в секунду

bool show_light_markers = true;

// Материал фигур: альбедо своё у каждой, блик общий.
float material_specular[3] = {1.0f, 1.0f, 1.0f};
float material_shininess = 48.0f;
float procedural_color_mix = 0.5f; // сколько процедурного цвета вершины в альбедо

bool show_floor = true;
float floor_albedo[3] = {0.45f, 0.45f, 0.5f};

Vec3 object_position = {0.0f, 0.4f, 0.0f};
Vec3 object_rotation = {0.0f, 0.0f, 0.0f}; // углы Эйлера в градусах
Vec3 object_scale = {1.0f, 1.0f, 1.0f};

bool auto_rotate = true;
float auto_rotate_speed = 30.0f; // градусов в секунду
float auto_rotate_angle = 0.0f;

// Движение по замкнутой кривой Лиссажу из первой лабораторной.
bool animation_playing = true;
float animation_time = 0.0f;
float animation_speed = 0.6f;
float trajectory_radius = 1.6f;
float trajectory_height = 0.5f;
float trajectory_depth = 1.1f;
int trajectory_lobes = 2;
bool trajectory_turn = true;

int object_count = 3;
ObjectParams objects[max_objects] = {
	{{0.9f, 0.35f, 0.25f}, 1.00f, 0.0f},
	{{0.3f, 0.75f, 0.9f},  0.60f, 2.1f},
	{{0.5f, 0.9f,  0.4f},  0.45f, 4.2f},
	{{0.9f, 0.8f,  0.3f},  0.35f, 1.0f},
};

double previous_time = 0.0;

// Данные, посчитанные в update() и записанные в буферы в render().
SceneUniforms scene_uniforms;
ObjectUniforms object_uniforms[max_object_sets];
PointLightData point_light_data[max_point_lights];
SpotLightData spot_light_data[max_spot_lights];

// Икосаэдр: каждая грань получает собственные три вершины (20 * 3 = 60), хотя
// геометрически вершин всего 12. Так у соседних граней в общей точке могут быть
// разные нормали и разные цвета.
//
// Нормали считаются процедурно: нормаль вершины -- это нормаль её грани.
// У выпуклого тела с центром в начале координат она совпадает с направлением на
// центр грани. Такие нормали дают плоскогранную заливку: каждая грань освещена
// равномерно, а блики ложатся по граням, а не как на сфере.
void buildIcosahedron(Vertex* out_vertices) {
	for (uint32_t face = 0; face < icosahedron_face_count; ++face) {
		Vec3 positions[3];
		Vec3 center = {0.0f, 0.0f, 0.0f};

		for (uint32_t i = 0; i < 3; ++i) {
			// Нормировка вписывает фигуру в единичную сферу.
			positions[i] = linalg::normalize(icosahedron_corners[icosahedron_faces[face * 3 + i]]);
			center = center + positions[i];
		}

		const Vec3 normal = linalg::normalize(center);

		for (uint32_t i = 0; i < 3; ++i) {
			const Vec3 direction = linalg::normalize(positions[i] * 0.6f + normal * 0.4f);
			const Vec3 color = {
				0.5f + 0.5f * direction.x,
				0.5f + 0.5f * direction.y,
				0.5f + 0.5f * direction.z,
			};

			out_vertices[face * 3 + i] = {positions[i], normal, color};
		}
	}
}

// Пол -- квадрат в плоскости XZ. На нём видно и затухание точечных источников,
// и мягкий край пятна прожектора.
void buildFloor(Vertex* out_vertices) {
	constexpr float size = 7.0f;
	constexpr float height = -1.2f;
	const Vec3 normal = {0.0f, 1.0f, 0.0f};
	const Vec3 color = {1.0f, 1.0f, 1.0f};

	const Vec3 corners[4] = {
		{-size, height, -size},
		{-size, height,  size},
		{ size, height,  size},
		{ size, height, -size},
	};

	const uint32_t order[floor_vertex_count] = {0, 1, 2, 0, 2, 3};
	for (uint32_t i = 0; i < floor_vertex_count; ++i) {
		out_vertices[i] = {corners[order[i]], normal, color};
	}
}

// Конус прожектора рисуется линиями. Он задан единичным: вершина в начале
// координат, ось вдоль +Z, обод радиуса 1 на расстоянии 1 (то есть угол 45
// градусов). Нужные длина и угол получаются растяжением в матрице модели.
void buildCone(Vertex* out_vertices) {
	const Vec3 normal = {0.0f, 0.0f, 1.0f};
	const Vec3 color = {1.0f, 1.0f, 1.0f};

	auto rim = [&](uint32_t index, uint32_t count) {
		const float angle = 2.0f * linalg::pi * static_cast<float>(index) /
		                    static_cast<float>(count);
		return Vec3{std::cos(angle), std::sin(angle), 1.0f};
	};

	uint32_t vertex = 0;

	// Лучи от вершины конуса к ободу.
	for (uint32_t i = 0; i < cone_ray_count; ++i) {
		out_vertices[vertex++] = {{0.0f, 0.0f, 0.0f}, normal, color};
		out_vertices[vertex++] = {rim(i, cone_ray_count), normal, color};
	}

	// Сам обод.
	for (uint32_t i = 0; i < cone_segment_count; ++i) {
		out_vertices[vertex++] = {rim(i, cone_segment_count), normal, color};
		out_vertices[vertex++] = {rim(i + 1, cone_segment_count), normal, color};
	}
}

bool createGeometry() {
	Vertex vertices[total_vertex_count];
	buildIcosahedron(vertices + icosahedron_first_vertex);
	buildFloor(vertices + floor_first_vertex);
	buildCone(vertices + cone_first_vertex);

	if (!graphics::createHostVisibleBuffer(sizeof(vertices),
	                                       VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
	                                       vertex_buffer)) {
		return false;
	}
	graphics::writeBuffer(vertex_buffer, vertices, sizeof(vertices));

	return true;
}

// Два набора дескрипторов. Набор 0 -- сцена: uniform-буфер с камерой,
// рассеянным и направленным светом плюс два storage-буфера с массивами
// источников. Набор 1 -- фигура: её матрицы и материал.
bool createDescriptors() {
	auto& context = graphics::internal::context;

	if (!graphics::createHostVisibleBuffer(sizeof(SceneUniforms),
	                                       VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
	                                       scene_uniform_buffer)) {
		return false;
	}

	// Storage-буферы выделяются на максимальное число источников, а сколько их
	// сейчас -- шейдер берёт из счётчиков в uniform-буфере сцены.
	if (!graphics::createHostVisibleBuffer(sizeof(PointLightData) * max_point_lights,
	                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
	                                       point_light_buffer)) {
		return false;
	}

	if (!graphics::createHostVisibleBuffer(sizeof(SpotLightData) * max_spot_lights,
	                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
	                                       spot_light_buffer)) {
		return false;
	}

	for (uint32_t i = 0; i < max_object_sets; ++i) {
		if (!graphics::createHostVisibleBuffer(sizeof(ObjectUniforms),
		                                       VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		                                       object_uniform_buffers[i])) {
			return false;
		}
	}

	const VkDescriptorSetLayoutBinding scene_bindings[] = {
		{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			.binding = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			.binding = 2,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
	};

	const VkDescriptorSetLayoutCreateInfo scene_set_layout = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 3,
		.pBindings = scene_bindings,
	};

	if (vkCreateDescriptorSetLayout(context.device, &scene_set_layout, nullptr,
	                                &vk_scene_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan descriptor set layout for scene data\n";
		return false;
	}

	const VkDescriptorSetLayoutBinding object_binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
	};

	const VkDescriptorSetLayoutCreateInfo object_set_layout = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &object_binding,
	};

	if (vkCreateDescriptorSetLayout(context.device, &object_set_layout, nullptr,
	                                &vk_object_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan descriptor set layout for object data\n";
		return false;
	}

	const VkDescriptorPoolSize pool_sizes[] = {
		{
			.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1 + max_object_sets,
		},
		{
			.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 2,
		},
	};

	const VkDescriptorPoolCreateInfo descriptor_pool = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 1 + max_object_sets,
		.poolSizeCount = 2,
		.pPoolSizes = pool_sizes,
	};

	if (vkCreateDescriptorPool(context.device, &descriptor_pool, nullptr,
	                           &vk_descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan descriptor pool\n";
		return false;
	}

	const VkDescriptorSetAllocateInfo scene_allocation = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = vk_descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &vk_scene_set_layout,
	};

	if (vkAllocateDescriptorSets(context.device, &scene_allocation, &vk_scene_set) != VK_SUCCESS) {
		std::cerr << "Failed to allocate Vulkan descriptor set for scene data\n";
		return false;
	}

	VkDescriptorSetLayout object_layouts[max_object_sets];
	for (uint32_t i = 0; i < max_object_sets; ++i) {
		object_layouts[i] = vk_object_set_layout;
	}

	const VkDescriptorSetAllocateInfo object_allocation = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = vk_descriptor_pool,
		.descriptorSetCount = max_object_sets,
		.pSetLayouts = object_layouts,
	};

	if (vkAllocateDescriptorSets(context.device, &object_allocation,
	                             vk_object_sets) != VK_SUCCESS) {
		std::cerr << "Failed to allocate Vulkan descriptor sets for objects\n";
		return false;
	}

	// Наборы связываются с буферами один раз: сами буферы потом не меняются,
	// меняется только их содержимое.
	VkDescriptorBufferInfo buffer_infos[3 + max_object_sets];
	VkWriteDescriptorSet writes[3 + max_object_sets];

	const graphics::Buffer* scene_buffers[3] = {
		&scene_uniform_buffer, &point_light_buffer, &spot_light_buffer,
	};

	for (uint32_t i = 0; i < 3; ++i) {
		buffer_infos[i] = {
			.buffer = scene_buffers[i]->handle,
			.offset = 0,
			.range = scene_buffers[i]->size,
		};
		writes[i] = {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = vk_scene_set,
			.dstBinding = i,
			.descriptorCount = 1,
			.descriptorType = (i == 0) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
			                           : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.pBufferInfo = &buffer_infos[i],
		};
	}

	for (uint32_t i = 0; i < max_object_sets; ++i) {
		buffer_infos[3 + i] = {
			.buffer = object_uniform_buffers[i].handle,
			.offset = 0,
			.range = sizeof(ObjectUniforms),
		};
		writes[3 + i] = {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = vk_object_sets[i],
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &buffer_infos[3 + i],
		};
	}

	vkUpdateDescriptorSets(context.device, 3 + max_object_sets, writes, 0, nullptr);

	return true;
}

// Конвейеры отличаются только шейдерами и типом примитива, всё остальное у них
// одинаковое, поэтому создаются они одной функцией.
bool createPipeline(const char* vertex_path, const char* fragment_path,
                    VkPrimitiveTopology topology, VkPipeline& out_pipeline) {
	auto& context = graphics::internal::context;

	VkShaderModule vertex_shader = graphics::loadShaderModule(vertex_path);
	VkShaderModule fragment_shader = graphics::loadShaderModule(fragment_path);
	if (vertex_shader == VK_NULL_HANDLE || fragment_shader == VK_NULL_HANDLE) {
		vkDestroyShaderModule(context.device, vertex_shader, nullptr);
		vkDestroyShaderModule(context.device, fragment_shader, nullptr);
		return false;
	}

	const VkPipelineShaderStageCreateInfo stages[] = {
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertex_shader,
			.pName = "main",
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragment_shader,
			.pName = "main",
		},
	};

	// Как читать вершинный буфер: одна привязка, три атрибута.
	const VkVertexInputBindingDescription vertex_binding = {
		.binding = 0,
		.stride = sizeof(Vertex),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};

	const VkVertexInputAttributeDescription vertex_attributes[] = {
		{
			.location = 0,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, position),
		},
		{
			.location = 1,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, normal),
		},
		{
			.location = 2,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, color),
		},
	};

	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &vertex_binding,
		.vertexAttributeDescriptionCount = 3,
		.pVertexAttributeDescriptions = vertex_attributes,
	};

	const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = topology,
	};

	// Размеры окна меняются, поэтому viewport и scissor -- динамические
	// состояния, они задаются командами при записи командного буфера.
	const VkDynamicState dynamic_states[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	};

	const VkPipelineDynamicStateCreateInfo dynamic_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = 2,
		.pDynamicStates = dynamic_states,
	};

	const VkPipelineViewportStateCreateInfo viewport_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};

	// Отсечение задних граней выключено: фигуру можно "растянуть"
	// отрицательным масштабом, что меняет ориентацию обхода треугольников,
	// а пол должен быть виден и снизу.
	const VkPipelineRasterizationStateCreateInfo rasterization = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_NONE,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.lineWidth = 1.0f,
	};

	const VkPipelineMultisampleStateCreateInfo multisample = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
		.minSampleShading = 1.0f,
	};

	// Тест глубины: ближний фрагмент (меньшее z) перекрывает дальний.
	const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS,
		.minDepthBounds = 0.0f,
		.maxDepthBounds = 1.0f,
	};

	const VkPipelineColorBlendAttachmentState color_blend_attachment = {
		.blendEnable = VK_FALSE,
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	};

	const VkPipelineColorBlendStateCreateInfo color_blend = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &color_blend_attachment,
	};

	const VkGraphicsPipelineCreateInfo pipeline = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = 2,
		.pStages = stages,
		.pVertexInputState = &vertex_input,
		.pInputAssemblyState = &input_assembly,
		.pViewportState = &viewport_state,
		.pRasterizationState = &rasterization,
		.pMultisampleState = &multisample,
		.pDepthStencilState = &depth_stencil,
		.pColorBlendState = &color_blend,
		.pDynamicState = &dynamic_state,
		.layout = vk_pipeline_layout,
		.renderPass = context.render_pass,
		.subpass = 0,
	};

	const bool ok = vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipeline,
	                                          nullptr, &out_pipeline) == VK_SUCCESS;

	// Модули нужны только во время создания конвейера.
	vkDestroyShaderModule(context.device, vertex_shader, nullptr);
	vkDestroyShaderModule(context.device, fragment_shader, nullptr);

	return ok;
}

bool createPipelines() {
	auto& context = graphics::internal::context;

	// Оба набора дескрипторов общие для всех конвейеров.
	const VkDescriptorSetLayout set_layouts[] = {
		vk_scene_set_layout,
		vk_object_set_layout,
	};

	const VkPipelineLayoutCreateInfo pipeline_layout = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 2,
		.pSetLayouts = set_layouts,
	};

	if (vkCreatePipelineLayout(context.device, &pipeline_layout, nullptr,
	                           &vk_pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan pipeline layout\n";
		return false;
	}

	return createPipeline("shaders/object.vert.spv", "shaders/object.frag.spv",
	                      VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, vk_object_pipeline) &&
	       createPipeline("shaders/light.vert.spv", "shaders/light.frag.spv",
	                      VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, vk_light_pipeline) &&
	       createPipeline("shaders/light.vert.spv", "shaders/light.frag.spv",
	                      VK_PRIMITIVE_TOPOLOGY_LINE_LIST, vk_line_pipeline);
}

// Направление взгляда камеры по двум углам: yaw поворачивает вокруг
// вертикальной оси, pitch наклоняет вверх-вниз.
Vec3 cameraForward() {
	const float yaw = linalg::radians(camera_yaw);
	const float pitch = linalg::radians(camera_pitch);

	return {
		 std::sin(yaw) * std::cos(pitch),
		 std::sin(pitch),
		-std::cos(yaw) * std::cos(pitch),
	};
}

Mat4 viewMatrix() {
	return linalg::lookAt(camera_position, camera_position + cameraForward(),
	                      {0.0f, 1.0f, 0.0f});
}

// Точка траектории: кривая Лиссажу в трёх измерениях.
Vec3 trajectoryPoint(float t) {
	return {
		trajectory_radius * std::cos(t),
		trajectory_height * std::sin(static_cast<float>(trajectory_lobes) * t),
		trajectory_depth * std::sin(t),
	};
}

// M = T * Rz * Ry * Rx * S -- сначала масштаб, затем повороты, затем перенос.
Mat4 modelMatrix(int object_index) {
	const ObjectParams& params = objects[object_index];
	const float time = animation_time + params.phase;

	Vec3 position = object_position;
	float extra_yaw = auto_rotate_angle;

	if (animation_playing || animation_time != 0.0f) {
		const Vec3 point = trajectoryPoint(time);
		position = position + point;

		if (trajectory_turn) {
			// Фигура доворачивается по направлению движения: направление --
			// это разность двух близких точек траектории.
			const Vec3 ahead = trajectoryPoint(time + 0.05f);
			const Vec3 velocity = ahead - point;
			extra_yaw += linalg::degrees(std::atan2(velocity.x, velocity.z));
		}
	}

	const Mat4 rotation = linalg::rotationZ(linalg::radians(object_rotation.z)) *
	                      linalg::rotationY(linalg::radians(object_rotation.y + extra_yaw)) *
	                      linalg::rotationX(linalg::radians(object_rotation.x));

	const Vec3 scale = object_scale * params.scale;

	return linalg::translation(position) * rotation * linalg::scaling(scale);
}

Mat4 projectionMatrix() {
	const auto& extent = graphics::internal::context.swapchain_extent;
	const float aspect = (extent.height > 0)
	                   ? static_cast<float>(extent.width) / static_cast<float>(extent.height)
	                   : 1.0f;

	constexpr float near_plane = 0.1f;
	constexpr float far_plane = 100.0f;

	if (use_perspective) {
		return linalg::perspective(linalg::radians(fov_degrees), aspect, near_plane, far_plane);
	}

	return linalg::orthographic(ortho_height * aspect, ortho_height, near_plane, far_plane);
}

// Заполняет набор дескрипторов фигуры: матрицы и материал.
void setObjectUniforms(uint32_t index, const Mat4& model, const float albedo[3],
                       float procedural_mix, bool lit) {
	ObjectUniforms& uniforms = object_uniforms[index];

	uniforms.model = model;
	uniforms.normal_matrix = linalg::normalMatrix(model);

	for (int i = 0; i < 3; ++i) {
		uniforms.albedo[i] = albedo[i];
		uniforms.specular[i] = lit ? material_specular[i] : 0.0f;
	}
	uniforms.albedo[3] = procedural_mix;
	uniforms.specular[3] = material_shininess;
}

// Запись команд отрисовки всей сцены.
void recordScene(VkCommandBuffer cmd) {
	auto& context = graphics::internal::context;

	const VkViewport viewport = {
		.x = 0.0f,
		.y = 0.0f,
		.width = static_cast<float>(context.swapchain_extent.width),
		.height = static_cast<float>(context.swapchain_extent.height),
		.minDepth = 0.0f,
		.maxDepth = 1.0f,
	};
	vkCmdSetViewport(cmd, 0, 1, &viewport);

	const VkRect2D scissor = {
		.offset = {0, 0},
		.extent = context.swapchain_extent,
	};
	vkCmdSetScissor(cmd, 0, 1, &scissor);

	// Набор 0 одинаков для всей сцены, поэтому привязывается один раз.
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline_layout,
	                        0, 1, &vk_scene_set, 0, nullptr);

	const VkDeviceSize vertex_buffer_offset = 0;
	vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer.handle, &vertex_buffer_offset);

	// 4. Освещённые фигуры: икосаэдры и пол.
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_object_pipeline);

	for (int i = 0; i < object_count; ++i) {
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline_layout,
		                        1, 1, &vk_object_sets[i], 0, nullptr);
		vkCmdDraw(cmd, icosahedron_vertex_count, 1, icosahedron_first_vertex, 0);
	}

	if (show_floor) {
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline_layout,
		                        1, 1, &vk_object_sets[floor_set_index], 0, nullptr);
		vkCmdDraw(cmd, floor_vertex_count, 1, floor_first_vertex, 0);
	}

	// 5. Значки источников: тот же икосаэдр, но конвейером без освещения.
	if (show_light_markers) {
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_light_pipeline);

		for (int i = 0; i < point_light_count; ++i) {
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline_layout,
			                        1, 1, &vk_object_sets[point_marker_set_index + i], 0, nullptr);
			vkCmdDraw(cmd, icosahedron_vertex_count, 1, icosahedron_first_vertex, 0);
		}

		for (int i = 0; i < spot_light_count; ++i) {
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline_layout,
			                        1, 1, &vk_object_sets[spot_marker_set_index + i], 0, nullptr);
			vkCmdDraw(cmd, icosahedron_vertex_count, 1, icosahedron_first_vertex, 0);
		}

		// Конусы показывают направление и внешний угол прожекторов.
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_line_pipeline);

		for (int i = 0; i < spot_light_count; ++i) {
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline_layout,
			                        1, 1, &vk_object_sets[cone_set_index + i], 0, nullptr);
			vkCmdDraw(cmd, cone_vertex_count, 1, cone_first_vertex, 0);
		}
	}

}

void drawInterface() {
	// Подписи интерфейса на латинице: шрифт ImGUI по умолчанию не содержит
	// глифов кириллицы.
	ImGui::Begin("Lab 2 / variant 11: regular icosahedron");

	ImGui::Text("%.1f FPS", static_cast<double>(ImGui::GetIO().Framerate));
	ImGui::Text("%d point and %d spot lights in storage buffers",
	            point_light_count, spot_light_count);

	if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::SliderFloat3("position", &camera_position.x, -12.0f, 12.0f, "%.2f");
		ImGui::SliderFloat("yaw", &camera_yaw, -180.0f, 180.0f, "%.0f deg");
		ImGui::SliderFloat("pitch", &camera_pitch, -89.0f, 89.0f, "%.0f deg");

		int projection = use_perspective ? 0 : 1;
		ImGui::RadioButton("perspective", &projection, 0);
		ImGui::SameLine();
		ImGui::RadioButton("orthographic", &projection, 1);
		use_perspective = (projection == 0);

		if (use_perspective) {
			ImGui::SliderFloat("field of view", &fov_degrees, 15.0f, 120.0f, "%.0f deg");
		} else {
			ImGui::SliderFloat("view height", &ortho_height, 0.5f, 12.0f, "%.2f");
		}

		if (ImGui::Button("Look at the scene")) {
			camera_position = {0.0f, 1.8f, 7.5f};
			camera_yaw = 0.0f;
			camera_pitch = -10.0f;
		}
	}

	if (ImGui::CollapsingHeader("Ambient and directional light", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::ColorEdit3("ambient colour", ambient_color);
		ImGui::SliderFloat("ambient intensity", &ambient_intensity, 0.0f, 1.0f, "%.2f");
		ImGui::SliderFloat3("direction to light", &light_direction.x, -1.0f, 1.0f, "%.2f");
		ImGui::ColorEdit3("sun colour", light_color);
		ImGui::SliderFloat("sun intensity", &light_intensity, 0.0f, 4.0f, "%.2f");
	}

	if (ImGui::CollapsingHeader("Point lights", ImGuiTreeNodeFlags_DefaultOpen)) {
		if (ImGui::Button("Add point light") && point_light_count < static_cast<int>(max_point_lights)) {
			point_lights[point_light_count] = {{0.0f, 1.5f, 0.0f}, {1.0f, 1.0f, 1.0f}, 10.0f};
			++point_light_count;
		}
		ImGui::SameLine();
		if (ImGui::Button("Remove point light") && point_light_count > 0) {
			--point_light_count;
		}

		for (int i = 0; i < point_light_count; ++i) {
			ImGui::PushID(100 + i);
			ImGui::Text("point light %d", i + 1);
			ImGui::SliderFloat3("position", point_lights[i].position, -8.0f, 8.0f, "%.2f");
			ImGui::ColorEdit3("colour", point_lights[i].color);
			ImGui::SliderFloat("intensity", &point_lights[i].intensity, 0.0f, 60.0f, "%.1f");
			ImGui::PopID();
		}
	}

	if (ImGui::CollapsingHeader("Spot lights", ImGuiTreeNodeFlags_DefaultOpen)) {
		if (ImGui::Button("Add spot light") && spot_light_count < static_cast<int>(max_spot_lights)) {
			spot_lights[spot_light_count] = {{0.0f, 3.0f, 0.0f}, {0.0f, -1.0f, 0.0f},
			                                 {1.0f, 1.0f, 1.0f}, 40.0f, 12.0f, 22.0f};
			++spot_light_count;
		}
		ImGui::SameLine();
		if (ImGui::Button("Remove spot light") && spot_light_count > 0) {
			--spot_light_count;
		}

		for (int i = 0; i < spot_light_count; ++i) {
			ImGui::PushID(200 + i);
			ImGui::Text("spot light %d", i + 1);
			ImGui::SliderFloat3("position", spot_lights[i].position, -8.0f, 8.0f, "%.2f");
			ImGui::SliderFloat3("direction", spot_lights[i].direction, -1.0f, 1.0f, "%.2f");
			ImGui::ColorEdit3("colour", spot_lights[i].color);
			ImGui::SliderFloat("intensity", &spot_lights[i].intensity, 0.0f, 200.0f, "%.1f");
			ImGui::SliderFloat("inner angle", &spot_lights[i].inner_angle, 1.0f, 80.0f, "%.0f deg");
			ImGui::SliderFloat("outer angle", &spot_lights[i].outer_angle, 1.0f, 80.0f, "%.0f deg");
			// Внешний угол меньше внутреннего смысла не имеет: край перестаёт
			// быть мягким.
			if (spot_lights[i].outer_angle < spot_lights[i].inner_angle) {
				spot_lights[i].outer_angle = spot_lights[i].inner_angle;
			}
			ImGui::PopID();
		}
	}

	if (ImGui::CollapsingHeader("Light visualisation", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::Checkbox("show light markers and cones", &show_light_markers);
		ImGui::Checkbox("animate point lights", &animate_lights);
		ImGui::SameLine();
		ImGui::SliderFloat("orbit deg/s", &light_orbit_speed, -180.0f, 180.0f, "%.0f");
	}

	if (ImGui::CollapsingHeader("Material", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::ColorEdit3("specular colour", material_specular);
		ImGui::SliderFloat("shininess", &material_shininess, 1.0f, 256.0f, "%.0f");
		ImGui::SliderFloat("procedural vertex colour", &procedural_color_mix, 0.0f, 1.0f, "%.2f");
		ImGui::Checkbox("show floor", &show_floor);
		ImGui::ColorEdit3("floor albedo", floor_albedo);
	}

	if (ImGui::CollapsingHeader("Objects")) {
		ImGui::SliderInt("object count", &object_count, 1, static_cast<int>(max_objects));
		ImGui::SliderFloat3("translation", &object_position.x, -3.0f, 3.0f, "%.2f");
		ImGui::SliderFloat3("rotation", &object_rotation.x, -180.0f, 180.0f, "%.0f deg");
		ImGui::SliderFloat3("scale", &object_scale.x, 0.1f, 3.0f, "%.2f");

		ImGui::Checkbox("spin", &auto_rotate);
		ImGui::SameLine();
		ImGui::SliderFloat("deg/s", &auto_rotate_speed, -180.0f, 180.0f, "%.0f");

		if (ImGui::Button(animation_playing ? "Pause" : "Play")) {
			animation_playing = !animation_playing;
		}
		ImGui::SameLine();
		if (ImGui::Button("Stop")) {
			animation_playing = false;
			animation_time = 0.0f;
		}
		ImGui::SliderFloat("animation speed", &animation_speed, 0.0f, 3.0f, "%.2f");
		ImGui::SliderFloat("trajectory radius", &trajectory_radius, 0.0f, 3.0f, "%.2f");
		ImGui::SliderFloat("trajectory height", &trajectory_height, 0.0f, 1.5f, "%.2f");
		ImGui::SliderFloat("trajectory depth", &trajectory_depth, 0.0f, 3.0f, "%.2f");
		ImGui::SliderInt("trajectory lobes", &trajectory_lobes, 1, 5);
		ImGui::Checkbox("turn along the path", &trajectory_turn);

		for (int i = 0; i < object_count; ++i) {
			ImGui::PushID(i);
			ImGui::Text("object %d", i + 1);
			ImGui::ColorEdit3("albedo", objects[i].albedo);
			ImGui::SliderFloat("size", &objects[i].scale, 0.1f, 1.5f, "%.2f");
			ImGui::SliderFloat("path offset", &objects[i].phase, 0.0f, 6.28f, "%.2f");
			ImGui::PopID();
		}
	}

	ImGui::End();
}

} // namespace

bool initialize() {
	if (!createGeometry() || !createDescriptors()) {
		return false;
	}

	return createPipelines();
}

void shutdown() {
	auto& context = graphics::internal::context;
	vkQueueWaitIdle(context.graphics_queue);

	vkDestroyPipeline(context.device, vk_line_pipeline, nullptr);
	vkDestroyPipeline(context.device, vk_light_pipeline, nullptr);
	vkDestroyPipeline(context.device, vk_object_pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, vk_pipeline_layout, nullptr);

	// Наборы дескрипторов уничтожаются вместе с пулом, из которого выделены.
	vkDestroyDescriptorPool(context.device, vk_descriptor_pool, nullptr);
	vkDestroyDescriptorSetLayout(context.device, vk_object_set_layout, nullptr);
	vkDestroyDescriptorSetLayout(context.device, vk_scene_set_layout, nullptr);

	for (uint32_t i = 0; i < max_object_sets; ++i) {
		graphics::destroyBuffer(object_uniform_buffers[i]);
	}
	graphics::destroyBuffer(spot_light_buffer);
	graphics::destroyBuffer(point_light_buffer);
	graphics::destroyBuffer(scene_uniform_buffer);

	graphics::destroyBuffer(vertex_buffer);
}

void update(double time) {
	const float delta_time = static_cast<float>(time - previous_time);
	previous_time = time;

	drawInterface();

	if (animation_playing) {
		animation_time += animation_speed * delta_time;
		if (animation_time > 2.0f * linalg::pi * 100.0f) {
			animation_time = 0.0f;
		}
	}

	if (auto_rotate) {
		auto_rotate_angle += auto_rotate_speed * delta_time;
		if (auto_rotate_angle > 360.0f || auto_rotate_angle < -360.0f) {
			auto_rotate_angle = std::fmod(auto_rotate_angle, 360.0f);
		}
	}

	if (animate_lights) {
		// Точечные источники кружат вокруг вертикальной оси сцены.
		const float angle = linalg::radians(light_orbit_speed * delta_time);
		const float s = std::sin(angle);
		const float c = std::cos(angle);

		for (int i = 0; i < point_light_count; ++i) {
			const float x = point_lights[i].position[0];
			const float z = point_lights[i].position[2];
			point_lights[i].position[0] = x * c + z * s;
			point_lights[i].position[2] = -x * s + z * c;
		}
	}

	// --- данные сцены ---
	scene_uniforms.view_projection = projectionMatrix() * viewMatrix();

	// Позиция камеры нужна фрагментному шейдеру: от неё считается направление
	// на наблюдателя, без которого не посчитать блик.
	scene_uniforms.camera_position[0] = camera_position.x;
	scene_uniforms.camera_position[1] = camera_position.y;
	scene_uniforms.camera_position[2] = camera_position.z;
	scene_uniforms.camera_position[3] = 1.0f;

	const Vec3 direction = linalg::normalize(light_direction);
	scene_uniforms.light_direction[0] = direction.x;
	scene_uniforms.light_direction[1] = direction.y;
	scene_uniforms.light_direction[2] = direction.z;
	scene_uniforms.light_direction[3] = 0.0f;

	for (int i = 0; i < 3; ++i) {
		scene_uniforms.ambient[i] = ambient_color[i];
		scene_uniforms.light_color[i] = light_color[i];
	}
	scene_uniforms.ambient[3] = ambient_intensity;
	scene_uniforms.light_color[3] = light_intensity;

	scene_uniforms.light_counts[0] = static_cast<uint32_t>(point_light_count);
	scene_uniforms.light_counts[1] = static_cast<uint32_t>(spot_light_count);
	scene_uniforms.light_counts[2] = 0;
	scene_uniforms.light_counts[3] = 0;

	// --- массивы источников ---
	for (int i = 0; i < point_light_count; ++i) {
		for (int k = 0; k < 3; ++k) {
			point_light_data[i].position[k] = point_lights[i].position[k];
			point_light_data[i].color[k] = point_lights[i].color[k];
		}
		point_light_data[i].position[3] = 1.0f;
		point_light_data[i].color[3] = point_lights[i].intensity;
	}

	for (int i = 0; i < spot_light_count; ++i) {
		const Vec3 spot_direction = linalg::normalize({spot_lights[i].direction[0],
		                                               spot_lights[i].direction[1],
		                                               spot_lights[i].direction[2]});
		for (int k = 0; k < 3; ++k) {
			spot_light_data[i].position[k] = spot_lights[i].position[k];
			spot_light_data[i].color[k] = spot_lights[i].color[k];
		}
		spot_light_data[i].position[3] = 1.0f;
		spot_light_data[i].direction[0] = spot_direction.x;
		spot_light_data[i].direction[1] = spot_direction.y;
		spot_light_data[i].direction[2] = spot_direction.z;
		spot_light_data[i].direction[3] = 0.0f;
		spot_light_data[i].color[3] = spot_lights[i].intensity;

		// В шейдер уходят косинусы половинных углов: сравнивать косинусы
		// дешевле, чем считать арккосинус для каждого фрагмента.
		spot_light_data[i].cutoff[0] = std::cos(linalg::radians(spot_lights[i].inner_angle));
		spot_light_data[i].cutoff[1] = std::cos(linalg::radians(spot_lights[i].outer_angle));
		spot_light_data[i].cutoff[2] = 0.0f;
		spot_light_data[i].cutoff[3] = 0.0f;
	}

	// --- данные фигур ---
	for (int i = 0; i < object_count; ++i) {
		setObjectUniforms(static_cast<uint32_t>(i), modelMatrix(i), objects[i].albedo,
		                  procedural_color_mix, true);
	}

	if (show_floor) {
		setObjectUniforms(floor_set_index, linalg::identity(), floor_albedo, 0.0f, true);
	}

	if (show_light_markers) {
		// Значок источника -- маленький икосаэдр его цвета. Он рисуется
		// отдельным конвейером без освещения, поэтому его альбедо и есть
		// итоговый цвет на экране.
		for (int i = 0; i < point_light_count; ++i) {
			const Vec3 position = {point_lights[i].position[0], point_lights[i].position[1],
			                       point_lights[i].position[2]};
			const Mat4 model = linalg::translation(position) *
			                   linalg::scaling({0.09f, 0.09f, 0.09f});

			setObjectUniforms(point_marker_set_index + static_cast<uint32_t>(i), model,
			                  point_lights[i].color, 0.0f, false);
		}

		for (int i = 0; i < spot_light_count; ++i) {
			const Vec3 position = {spot_lights[i].position[0], spot_lights[i].position[1],
			                       spot_lights[i].position[2]};
			const Mat4 marker = linalg::translation(position) *
			                    linalg::scaling({0.09f, 0.09f, 0.09f});

			setObjectUniforms(spot_marker_set_index + static_cast<uint32_t>(i), marker,
			                  spot_lights[i].color, 0.0f, false);

			// Конус задан единичным (угол 45 градусов, длина 1), нужные угол и
			// длина получаются растяжением: радиус обода -- тангенс внешнего угла.
			constexpr float cone_length = 4.0f;
			const float radius = std::tan(linalg::radians(spot_lights[i].outer_angle)) * cone_length;

			const Vec3 spot_direction = {spot_lights[i].direction[0], spot_lights[i].direction[1],
			                             spot_lights[i].direction[2]};
			const Mat4 cone = linalg::translation(position) *
			                  linalg::orientation(spot_direction) *
			                  linalg::scaling({radius, radius, cone_length});

			const float white[3] = {1.0f, 1.0f, 1.0f};
			setObjectUniforms(cone_set_index + static_cast<uint32_t>(i), cone, white, 0.0f, false);
		}
	}
}

void render(const graphics::internal::FrameData& fd) {
	auto& context = graphics::internal::context;
	VkCommandBuffer cmd = fd.command_buffer;
	if (cmd == VK_NULL_HANDLE)
		return; // the frame could not be prepared

	// 1. Start recording commands
	const VkCommandBufferBeginInfo begin_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vkBeginCommandBuffer(cmd, &begin_info);

	// Данные кадра пишутся здесь, а не в update(): к этому моменту
	// graphics::internal::prepare() уже дождался, пока видеокарта закончит
	// предыдущий кадр, поэтому эти буферы можно безопасно перезаписывать.
	graphics::writeBuffer(scene_uniform_buffer, &scene_uniforms, sizeof(scene_uniforms));
	graphics::writeBuffer(point_light_buffer, point_light_data, sizeof(point_light_data));
	graphics::writeBuffer(spot_light_buffer, spot_light_data, sizeof(spot_light_data));

	for (uint32_t i = 0; i < max_object_sets; ++i) {
		graphics::writeBuffer(object_uniform_buffers[i], &object_uniforms[i],
		                      sizeof(ObjectUniforms));
	}

	// 2. The values used to clear the color and depth images
	const VkClearValue clear_values[] = {
		{.color = {.float32 = {0.04f, 0.04f, 0.06f, 1.0f}}}, // почти чёрный
		{.depthStencil = {.depth = 1.0f, .stencil = 0}},     // depth 1.0 = "far away"
	};

	// 3. Begin the render pass (this clears the screen)
	const VkRenderPassBeginInfo render_pass_begin = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = context.render_pass,
		.framebuffer = fd.framebuffer,
		.renderArea = {.extent = context.swapchain_extent},
		.clearValueCount = 2,
		.pClearValues = clear_values,
	};
	vkCmdBeginRenderPass(cmd, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);

	recordScene(cmd);

	// 6. Finish
	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

} // namespace application
