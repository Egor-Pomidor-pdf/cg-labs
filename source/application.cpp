// Лабораторная работа №1 -- Основы 3D-графики
// Вариант 11: правильный икосаэдр
//
// Икосаэдр задаётся таблицей из 12 вершин и таблицей из 20 треугольных граней
// и рисуется в реальном времени с тестом глубины. Матрица model-view-projection
// пересчитывается каждый кадр на CPU и передаётся в вершинный шейдер через
// push-константы.

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

struct Vertex {
	Vec3 position;
	Vec3 color;
};

// Данные сцены: одинаковы для всех объектов кадра. Лежат в наборе
// дескрипторов 0, который привязывается один раз за кадр.
struct SceneUniforms {
	Mat4 view;
	Mat4 projection;
};

// Данные одного объекта. Лежат в наборе дескрипторов 1 -- у каждой фигуры он
// свой, и перед её отрисовкой привязывается именно он.
struct ObjectUniforms {
	Mat4 model;
	float tint[4]; // цвет из интерфейса, на него домножается цвет вершины
};

// Сколько фигур можно нарисовать одновременно: столько же наборов
// дескрипторов выделяется при запуске.
constexpr uint32_t max_objects = 4;

// Параметры фигуры, которыми управляет интерфейс.
struct ObjectParams {
	float tint[3];
	float scale;  // множитель к общему растяжению
	float phase;  // сдвиг по траектории, радианы
};

constexpr uint32_t icosahedron_corner_count = 12; // уникальных вершин тела
constexpr uint32_t icosahedron_face_count = 20;   // треугольных граней
constexpr uint32_t icosahedron_vertex_count = icosahedron_face_count * 3;

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

VkPipelineLayout vk_pipeline_layout = VK_NULL_HANDLE;
VkPipeline vk_pipeline = VK_NULL_HANDLE;

// Параметры сцены, которыми управляет интерфейс.
bool use_perspective = true;
float fov_degrees = 60.0f;
float ortho_height = 1.4f;    // половина высоты области видимости
float camera_distance = 3.0f;

Vec3 object_position = {0.0f, 0.0f, 0.0f};
Vec3 object_rotation = {0.0f, 0.0f, 0.0f}; // углы Эйлера в градусах
Vec3 object_scale = {1.0f, 1.0f, 1.0f};

bool auto_rotate = true;
float auto_rotate_speed = 30.0f; // градусов в секунду
float auto_rotate_angle = 0.0f;

// Движение по траектории: фигура летит по замкнутой кривой Лиссажу и
// одновременно поворачивается. Пауза, скорость и параметры кривой задаются
// в интерфейсе.
bool animation_playing = true;
float animation_time = 0.0f;      // параметр кривой, растёт только при проигрывании
float animation_speed = 0.6f;     // во сколько раз быстрее базовой скорости
float trajectory_radius = 0.9f;   // радиус по горизонтали
float trajectory_height = 0.45f;  // амплитуда по вертикали
float trajectory_depth = 0.6f;    // амплитуда по глубине
int trajectory_lobes = 2;         // число лепестков кривой по вертикали
bool trajectory_turn = true;      // доворачивать фигуру по ходу движения

// Фигуры на сцене. Все летят по одной траектории, но с разным сдвигом,
// размером и цветом.
int object_count = 3;
ObjectParams objects[max_objects] = {
	{{1.0f, 1.0f, 1.0f}, 1.00f, 0.0f},
	{{1.0f, 0.5f, 0.3f}, 0.60f, 2.1f},
	{{0.4f, 0.8f, 1.0f}, 0.45f, 4.2f},
	{{0.6f, 1.0f, 0.5f}, 0.35f, 1.0f},
};

double previous_time = 0.0;

// Данные, посчитанные в update() и записанные в буферы в render().
SceneUniforms scene_uniforms;
ObjectUniforms object_uniforms[max_objects];

// Наборы дескрипторов и буферы под них.
graphics::Buffer scene_uniform_buffer;
graphics::Buffer object_uniform_buffers[max_objects];

VkDescriptorSetLayout vk_scene_set_layout = VK_NULL_HANDLE;
VkDescriptorSetLayout vk_object_set_layout = VK_NULL_HANDLE;
VkDescriptorPool vk_descriptor_pool = VK_NULL_HANDLE;
VkDescriptorSet vk_scene_set = VK_NULL_HANDLE;
VkDescriptorSet vk_object_sets[max_objects] = {};

// Вершинный буфер строится из таблицы координат и таблицы граней.
//
// Каждая грань получает собственные три вершины (20 * 3 = 60 штук), хотя
// геометрически вершин всего 12. Это нужно, чтобы соседние грани могли задать
// разный цвет в общей точке.
//
// Цвет вершины назначается процедурно -- по её позиции в локальной системе
// координат фигуры, с добавкой направления на центр грани. Без этой добавки
// соседние грани получали бы в общих вершинах одинаковый цвет, переходы между
// гранями исчезли бы, и без освещения фигура выглядела бы однотонным пятном.
// В шейдере этот цвет умножается на цвет, выбранный в интерфейсе.
void buildVertices(Vertex (&out_vertices)[icosahedron_vertex_count]) {
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

			out_vertices[face * 3 + i] = {positions[i], color};
		}
	}
}

bool createGeometry() {
	Vertex vertices[icosahedron_vertex_count];
	buildVertices(vertices);

	if (!graphics::createHostVisibleBuffer(sizeof(vertices),
	                                       VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
	                                       vertex_buffer)) {
		return false;
	}
	graphics::writeBuffer(vertex_buffer, vertices, sizeof(vertices));

	return true;
}

// Два набора дескрипторов: общий для сцены и отдельный для каждой фигуры.
// Благодаря этому view и projection записываются один раз за кадр, а перед
// отрисовкой каждой фигуры меняется только её собственный набор.
bool createDescriptors() {
	auto& context = graphics::internal::context;

	if (!graphics::createHostVisibleBuffer(sizeof(SceneUniforms),
	                                       VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
	                                       scene_uniform_buffer)) {
		return false;
	}

	for (uint32_t i = 0; i < max_objects; ++i) {
		if (!graphics::createHostVisibleBuffer(sizeof(ObjectUniforms),
		                                       VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		                                       object_uniform_buffers[i])) {
			return false;
		}
	}

	// Описание набора 0: один буфер, который читает вершинный шейдер.
	const VkDescriptorSetLayoutBinding scene_binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
	};

	const VkDescriptorSetLayoutCreateInfo scene_set_layout = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &scene_binding,
	};

	if (vkCreateDescriptorSetLayout(context.device, &scene_set_layout, nullptr,
	                                &vk_scene_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan descriptor set layout for scene data\n";
		return false;
	}

	// Описание набора 1: цвет нужен и фрагментному шейдеру тоже.
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

	// Пул, из которого выделяются наборы: один для сцены и по одному на фигуру.
	const VkDescriptorPoolSize pool_size = {
		.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1 + max_objects,
	};

	const VkDescriptorPoolCreateInfo descriptor_pool = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 1 + max_objects,
		.poolSizeCount = 1,
		.pPoolSizes = &pool_size,
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

	VkDescriptorSetLayout object_layouts[max_objects];
	for (uint32_t i = 0; i < max_objects; ++i) {
		object_layouts[i] = vk_object_set_layout;
	}

	const VkDescriptorSetAllocateInfo object_allocation = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = vk_descriptor_pool,
		.descriptorSetCount = max_objects,
		.pSetLayouts = object_layouts,
	};

	if (vkAllocateDescriptorSets(context.device, &object_allocation,
	                             vk_object_sets) != VK_SUCCESS) {
		std::cerr << "Failed to allocate Vulkan descriptor sets for objects\n";
		return false;
	}

	// Наборы связываются с буферами один раз: сами буферы потом не меняются,
	// меняется только их содержимое.
	VkDescriptorBufferInfo buffer_infos[1 + max_objects];
	VkWriteDescriptorSet writes[1 + max_objects];

	buffer_infos[0] = {
		.buffer = scene_uniform_buffer.handle,
		.offset = 0,
		.range = sizeof(SceneUniforms),
	};
	writes[0] = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = vk_scene_set,
		.dstBinding = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.pBufferInfo = &buffer_infos[0],
	};

	for (uint32_t i = 0; i < max_objects; ++i) {
		buffer_infos[1 + i] = {
			.buffer = object_uniform_buffers[i].handle,
			.offset = 0,
			.range = sizeof(ObjectUniforms),
		};
		writes[1 + i] = {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = vk_object_sets[i],
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &buffer_infos[1 + i],
		};
	}

	vkUpdateDescriptorSets(context.device, 1 + max_objects, writes, 0, nullptr);

	return true;
}

bool createPipeline() {
	auto& context = graphics::internal::context;

	VkShaderModule vertex_shader = graphics::loadShaderModule("shaders/object.vert.spv");
	VkShaderModule fragment_shader = graphics::loadShaderModule("shaders/object.frag.spv");
	if (vertex_shader == VK_NULL_HANDLE || fragment_shader == VK_NULL_HANDLE) {
		vkDestroyShaderModule(context.device, vertex_shader, nullptr);
		vkDestroyShaderModule(context.device, fragment_shader, nullptr);
		return false;
	}

	// Конвейер ожидает два набора дескрипторов: нулевой -- данные сцены,
	// первый -- данные конкретной фигуры.
	const VkDescriptorSetLayout set_layouts[] = {
		vk_scene_set_layout,
		vk_object_set_layout,
	};

	const VkPipelineLayoutCreateInfo pipeline_layout = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 2,
		.pSetLayouts = set_layouts,
	};

	bool ok = vkCreatePipelineLayout(context.device, &pipeline_layout, nullptr,
	                                 &vk_pipeline_layout) == VK_SUCCESS;
	if (ok) {
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

		// Как читать вершинный буфер: одна привязка, два атрибута.
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
				.offset = offsetof(Vertex, color),
			},
		};

		const VkPipelineVertexInputStateCreateInfo vertex_input = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			.vertexBindingDescriptionCount = 1,
			.pVertexBindingDescriptions = &vertex_binding,
			.vertexAttributeDescriptionCount = 2,
			.pVertexAttributeDescriptions = vertex_attributes,
		};

		const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
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
		// а видимость выпуклого тела и так корректно решает буфер глубины.
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

		ok = vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipeline,
		                               nullptr, &vk_pipeline) == VK_SUCCESS;
	}

	// Модули нужны только во время создания конвейера.
	vkDestroyShaderModule(context.device, vertex_shader, nullptr);
	vkDestroyShaderModule(context.device, fragment_shader, nullptr);

	return ok;
}

// Точка траектории: кривая Лиссажу в трёх измерениях. При lobes = 1 это
// наклонный эллипс, при lobes = 2 -- "восьмёрка", дальше -- более сложные петли.
Vec3 trajectoryPoint(float t) {
	return {
		trajectory_radius * std::cos(t),
		trajectory_height * std::sin(static_cast<float>(trajectory_lobes) * t),
		trajectory_depth * std::sin(t),
	};
}

// M = T * Rz * Ry * Rx * S -- сначала масштаб, затем повороты, затем перенос.
// Каждая фигура идёт по той же траектории со своим сдвигом и размером.
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

void drawInterface() {
	// Подписи интерфейса на латинице: шрифт ImGUI по умолчанию не содержит
	// глифов кириллицы.
	ImGui::Begin("Lab 1 / variant 11: regular icosahedron");

	ImGui::Text("%.1f FPS", static_cast<double>(ImGui::GetIO().Framerate));
	ImGui::Text("%u faces, %u vertices in the buffer (%u distinct corners)",
	            icosahedron_face_count, icosahedron_vertex_count, icosahedron_corner_count);
	ImGui::Text("%d objects, %d descriptor sets bound per frame", object_count, object_count + 1);

	ImGui::SeparatorText("Projection");
	int projection = use_perspective ? 0 : 1;
	ImGui::RadioButton("perspective", &projection, 0);
	ImGui::SameLine();
	ImGui::RadioButton("orthographic", &projection, 1);
	use_perspective = (projection == 0);

	if (use_perspective) {
		ImGui::SliderFloat("field of view", &fov_degrees, 15.0f, 120.0f, "%.0f deg");
	} else {
		ImGui::SliderFloat("view height", &ortho_height, 0.5f, 8.0f, "%.2f");
	}
	ImGui::SliderFloat("camera distance", &camera_distance, 2.0f, 20.0f, "%.2f");

	ImGui::SeparatorText("Affine transformations");
	ImGui::SliderFloat3("translation", &object_position.x, -3.0f, 3.0f, "%.2f");
	ImGui::SliderFloat3("rotation", &object_rotation.x, -180.0f, 180.0f, "%.0f deg");
	ImGui::SliderFloat3("scale", &object_scale.x, 0.1f, 3.0f, "%.2f");

	ImGui::Checkbox("spin", &auto_rotate);
	ImGui::SameLine();
	ImGui::SliderFloat("deg/s", &auto_rotate_speed, -180.0f, 180.0f, "%.0f");

	ImGui::SeparatorText("Animation");
	if (ImGui::Button(animation_playing ? "Pause" : "Play")) {
		animation_playing = !animation_playing;
	}
	ImGui::SameLine();
	if (ImGui::Button("Stop")) {
		animation_playing = false;
		animation_time = 0.0f;
	}
	ImGui::SliderFloat("animation speed", &animation_speed, 0.0f, 3.0f, "%.2f");
	ImGui::SliderFloat("trajectory radius", &trajectory_radius, 0.0f, 2.0f, "%.2f");
	ImGui::SliderFloat("trajectory height", &trajectory_height, 0.0f, 1.5f, "%.2f");
	ImGui::SliderFloat("trajectory depth", &trajectory_depth, 0.0f, 2.0f, "%.2f");
	ImGui::SliderInt("trajectory lobes", &trajectory_lobes, 1, 5);
	ImGui::Checkbox("turn along the path", &trajectory_turn);

	ImGui::SeparatorText("Objects");
	ImGui::SliderInt("object count", &object_count, 1, static_cast<int>(max_objects));
	ImGui::TextWrapped("Every object has its own descriptor set with its own model "
	                   "matrix and colour; vertex colours are generated from vertex "
	                   "positions and multiplied by that colour in the shader.");

	for (int i = 0; i < object_count; ++i) {
		ImGui::PushID(i);
		ImGui::Text("object %d", i + 1);
		ImGui::ColorEdit3("tint", objects[i].tint);
		ImGui::SliderFloat("size", &objects[i].scale, 0.1f, 1.5f, "%.2f");
		ImGui::SliderFloat("path offset", &objects[i].phase, 0.0f, 6.28f, "%.2f");
		ImGui::PopID();
	}

	ImGui::Separator();
	if (ImGui::Button("Reset")) {
		object_position = {0.0f, 0.0f, 0.0f};
		object_rotation = {0.0f, 0.0f, 0.0f};
		object_scale = {1.0f, 1.0f, 1.0f};
		auto_rotate_angle = 0.0f;
		animation_time = 0.0f;
	}

	ImGui::End();
}

} // namespace

bool initialize() {
	if (!createGeometry() || !createDescriptors()) {
		return false;
	}

	return createPipeline();
}

void shutdown() {
	auto& context = graphics::internal::context;
	vkQueueWaitIdle(context.graphics_queue);

	vkDestroyPipeline(context.device, vk_pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, vk_pipeline_layout, nullptr);

	// Наборы дескрипторов уничтожаются вместе с пулом, из которого выделены.
	vkDestroyDescriptorPool(context.device, vk_descriptor_pool, nullptr);
	vkDestroyDescriptorSetLayout(context.device, vk_object_set_layout, nullptr);
	vkDestroyDescriptorSetLayout(context.device, vk_scene_set_layout, nullptr);

	for (uint32_t i = 0; i < max_objects; ++i) {
		graphics::destroyBuffer(object_uniform_buffers[i]);
	}
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

	// Камера смотрит в начало координат с положительной полуоси Z.
	scene_uniforms.view = linalg::lookAt({0.0f, 0.0f, camera_distance},
	                                     {0.0f, 0.0f, 0.0f},
	                                     {0.0f, 1.0f, 0.0f});
	scene_uniforms.projection = projectionMatrix();

	for (int i = 0; i < object_count; ++i) {
		object_uniforms[i].model = modelMatrix(i);
		object_uniforms[i].tint[0] = objects[i].tint[0];
		object_uniforms[i].tint[1] = objects[i].tint[1];
		object_uniforms[i].tint[2] = objects[i].tint[2];
		object_uniforms[i].tint[3] = 1.0f;
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
	for (int i = 0; i < object_count; ++i) {
		graphics::writeBuffer(object_uniform_buffers[i], &object_uniforms[i],
		                      sizeof(ObjectUniforms));
	}

	// 2. The values used to clear the color and depth images
	const VkClearValue clear_values[] = {
		{.color = {.float32 = {0.1f, 0.1f, 0.15f, 1.0f}}}, // dark blue-grey
		{.depthStencil = {.depth = 1.0f, .stencil = 0}},   // depth 1.0 = "far away"
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

	// 4. Draw the icosahedron
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline);

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

	// Фигуры делят одну и ту же геометрию: перед каждой меняется только её
	// собственный набор дескрипторов с матрицей модели и цветом.
	for (int i = 0; i < object_count; ++i) {
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline_layout,
		                        1, 1, &vk_object_sets[i], 0, nullptr);

		vkCmdDraw(cmd, icosahedron_vertex_count, 1, 0, 0);
	}

	// 5. Finish
	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

} // namespace application
