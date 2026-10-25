// Лабораторная работа №2 -- Камера и освещение
// Вариант 11: правильный икосаэдр
//
// Базовая часть: фигура из первой лабораторной рисуется относительно камеры,
// которой можно задать позицию, поворот и угол обзора, и освещается одним
// направленным источником по модели Блинн-Фонга. Параметры камеры, источника,
// рассеянного света и материала задаются из интерфейса.

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
struct Vertex {
	Vec3 position;
	Vec3 normal;
};

// Внешние данные шейдеров. Все поля выровнены по 16 байт (vec4 вместо vec3),
// как того требует раскладка std140 для uniform-буфера.
struct Uniforms {
	Mat4 model;
	Mat4 view_projection;
	Mat4 normal_matrix;
	float camera_position[4];
	float light_direction[4]; // направление НА источник
	float light_color[4];     // rgb -- цвет, w -- интенсивность
	float ambient[4];         // rgb -- цвет, w -- интенсивность
	float albedo[4];
	float specular[4];        // rgb -- цвет блика, w -- коэффициент блика
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
graphics::Buffer uniform_buffer;

VkDescriptorSetLayout vk_set_layout = VK_NULL_HANDLE;
VkDescriptorPool vk_descriptor_pool = VK_NULL_HANDLE;
VkDescriptorSet vk_descriptor_set = VK_NULL_HANDLE;

VkPipelineLayout vk_pipeline_layout = VK_NULL_HANDLE;
VkPipeline vk_pipeline = VK_NULL_HANDLE;

// Камера: позиция в мире и два угла поворота. При нулевых углах она смотрит
// вдоль -Z, то есть из точки {0, 0, 3} -- в начало координат.
Vec3 camera_position = {0.0f, 0.0f, 3.0f};
float camera_yaw = 0.0f;   // поворот вокруг вертикальной оси, градусы
float camera_pitch = 0.0f; // наклон вверх-вниз, градусы
float fov_degrees = 60.0f;

bool use_perspective = true;
float ortho_height = 1.4f; // половина высоты области видимости

// Направленный источник: светит с бесконечности, поэтому задаётся только
// направлением. В интерфейсе задаётся направление НА источник.
Vec3 light_direction = {-0.4f, 0.8f, 0.5f};
float light_color[3] = {1.0f, 0.95f, 0.85f};
float light_intensity = 1.0f;

// Рассеянный свет -- приближение освещения, пришедшего со всех сторон.
float ambient_color[3] = {0.45f, 0.55f, 0.8f};
float ambient_intensity = 0.15f;

// Материал по модели Блинн-Фонга.
float material_albedo[3] = {0.75f, 0.3f, 0.22f};
float material_specular[3] = {1.0f, 1.0f, 1.0f};
float material_shininess = 48.0f;

Vec3 object_position = {0.0f, 0.0f, 0.0f};
Vec3 object_rotation = {0.0f, 0.0f, 0.0f}; // углы Эйлера в градусах
Vec3 object_scale = {1.0f, 1.0f, 1.0f};

bool auto_rotate = true;
float auto_rotate_speed = 30.0f; // градусов в секунду
float auto_rotate_angle = 0.0f;

double previous_time = 0.0;

// Данные, посчитанные в update() и записанные в буфер в render().
Uniforms uniforms;

// Вершинный буфер строится из таблицы координат и таблицы граней.
//
// Каждая грань получает собственные три вершины (20 * 3 = 60 штук), хотя
// геометрически вершин всего 12. Так у соседних граней в общей точке могут быть
// разные нормали.
//
// Нормали считаются процедурно: нормаль вершины -- это нормаль её грани.
// У выпуклого тела с центром в начале координат она совпадает с направлением
// на центр грани. Такие нормали дают плоскогранную заливку: каждая грань
// освещена равномерно, а блики ложатся по граням, а не как на сфере.
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
			out_vertices[face * 3 + i] = {positions[i], normal};
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

// Параметры камеры, света и материала не влезают в push-константы
// (гарантированы 128 байт), поэтому они передаются через uniform-буфер.
bool createDescriptors() {
	auto& context = graphics::internal::context;

	if (!graphics::createHostVisibleBuffer(sizeof(Uniforms),
	                                       VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
	                                       uniform_buffer)) {
		return false;
	}

	// Буфер читают оба шейдера: вершинный -- матрицы, фрагментный -- свет.
	const VkDescriptorSetLayoutBinding binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
	};

	const VkDescriptorSetLayoutCreateInfo set_layout = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &binding,
	};

	if (vkCreateDescriptorSetLayout(context.device, &set_layout, nullptr,
	                                &vk_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan descriptor set layout\n";
		return false;
	}

	const VkDescriptorPoolSize pool_size = {
		.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
	};

	const VkDescriptorPoolCreateInfo descriptor_pool = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 1,
		.poolSizeCount = 1,
		.pPoolSizes = &pool_size,
	};

	if (vkCreateDescriptorPool(context.device, &descriptor_pool, nullptr,
	                           &vk_descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan descriptor pool\n";
		return false;
	}

	const VkDescriptorSetAllocateInfo allocation = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = vk_descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &vk_set_layout,
	};

	if (vkAllocateDescriptorSets(context.device, &allocation, &vk_descriptor_set) != VK_SUCCESS) {
		std::cerr << "Failed to allocate Vulkan descriptor set\n";
		return false;
	}

	// Набор связывается с буфером один раз: сам буфер потом не меняется,
	// меняется только его содержимое.
	const VkDescriptorBufferInfo buffer_info = {
		.buffer = uniform_buffer.handle,
		.offset = 0,
		.range = sizeof(Uniforms),
	};

	const VkWriteDescriptorSet write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = vk_descriptor_set,
		.dstBinding = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.pBufferInfo = &buffer_info,
	};

	vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);

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

	const VkPipelineLayoutCreateInfo pipeline_layout = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &vk_set_layout,
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
				.offset = offsetof(Vertex, normal),
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
	const Vec3 forward = cameraForward();
	return linalg::lookAt(camera_position, camera_position + forward, {0.0f, 1.0f, 0.0f});
}

// M = T * Rz * Ry * Rx * S -- сначала масштаб, затем повороты, затем перенос.
Mat4 modelMatrix() {
	const Mat4 rotation = linalg::rotationZ(linalg::radians(object_rotation.z)) *
	                      linalg::rotationY(linalg::radians(object_rotation.y + auto_rotate_angle)) *
	                      linalg::rotationX(linalg::radians(object_rotation.x));

	return linalg::translation(object_position) * rotation * linalg::scaling(object_scale);
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
	ImGui::Begin("Lab 2 / variant 11: regular icosahedron");

	ImGui::Text("%.1f FPS", static_cast<double>(ImGui::GetIO().Framerate));
	ImGui::Text("%u faces, %u vertices with procedural normals",
	            icosahedron_face_count, icosahedron_vertex_count);

	ImGui::SeparatorText("Camera");
	ImGui::SliderFloat3("position", &camera_position.x, -10.0f, 10.0f, "%.2f");
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
		ImGui::SliderFloat("view height", &ortho_height, 0.5f, 8.0f, "%.2f");
	}

	if (ImGui::Button("Look at the object")) {
		// Камера отъезжает по +Z и смотрит в начало координат.
		camera_position = {0.0f, 0.0f, 3.0f};
		camera_yaw = 0.0f;
		camera_pitch = 0.0f;
	}

	ImGui::SeparatorText("Directional light");
	ImGui::SliderFloat3("direction to light", &light_direction.x, -1.0f, 1.0f, "%.2f");
	ImGui::ColorEdit3("light colour", light_color);
	ImGui::SliderFloat("light intensity", &light_intensity, 0.0f, 4.0f, "%.2f");

	ImGui::SeparatorText("Ambient light");
	ImGui::ColorEdit3("ambient colour", ambient_color);
	ImGui::SliderFloat("ambient intensity", &ambient_intensity, 0.0f, 1.0f, "%.2f");

	ImGui::SeparatorText("Material");
	ImGui::ColorEdit3("albedo", material_albedo);
	ImGui::ColorEdit3("specular colour", material_specular);
	ImGui::SliderFloat("shininess", &material_shininess, 1.0f, 256.0f, "%.0f");

	ImGui::SeparatorText("Object");
	ImGui::SliderFloat3("translation", &object_position.x, -3.0f, 3.0f, "%.2f");
	ImGui::SliderFloat3("rotation", &object_rotation.x, -180.0f, 180.0f, "%.0f deg");
	ImGui::SliderFloat3("scale", &object_scale.x, 0.1f, 3.0f, "%.2f");

	ImGui::Checkbox("spin", &auto_rotate);
	ImGui::SameLine();
	ImGui::SliderFloat("deg/s", &auto_rotate_speed, -180.0f, 180.0f, "%.0f");

	if (ImGui::Button("Reset")) {
		object_position = {0.0f, 0.0f, 0.0f};
		object_rotation = {0.0f, 0.0f, 0.0f};
		object_scale = {1.0f, 1.0f, 1.0f};
		auto_rotate_angle = 0.0f;
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

	// Набор дескрипторов уничтожается вместе с пулом, из которого выделен.
	vkDestroyDescriptorPool(context.device, vk_descriptor_pool, nullptr);
	vkDestroyDescriptorSetLayout(context.device, vk_set_layout, nullptr);

	graphics::destroyBuffer(uniform_buffer);
	graphics::destroyBuffer(vertex_buffer);
}

void update(double time) {
	const float delta_time = static_cast<float>(time - previous_time);
	previous_time = time;

	drawInterface();

	if (auto_rotate) {
		auto_rotate_angle += auto_rotate_speed * delta_time;
		if (auto_rotate_angle > 360.0f || auto_rotate_angle < -360.0f) {
			auto_rotate_angle = std::fmod(auto_rotate_angle, 360.0f);
		}
	}

	const Mat4 model = modelMatrix();

	uniforms.model = model;
	uniforms.view_projection = projectionMatrix() * viewMatrix();
	uniforms.normal_matrix = linalg::normalMatrix(model);

	// Позиция камеры нужна фрагментному шейдеру: от неё считается направление
	// на наблюдателя, без которого не посчитать блик.
	uniforms.camera_position[0] = camera_position.x;
	uniforms.camera_position[1] = camera_position.y;
	uniforms.camera_position[2] = camera_position.z;
	uniforms.camera_position[3] = 1.0f;

	const Vec3 direction = linalg::normalize(light_direction);
	uniforms.light_direction[0] = direction.x;
	uniforms.light_direction[1] = direction.y;
	uniforms.light_direction[2] = direction.z;
	uniforms.light_direction[3] = 0.0f;

	for (int i = 0; i < 3; ++i) {
		uniforms.light_color[i] = light_color[i];
		uniforms.ambient[i] = ambient_color[i];
		uniforms.albedo[i] = material_albedo[i];
		uniforms.specular[i] = material_specular[i];
	}
	uniforms.light_color[3] = light_intensity;
	uniforms.ambient[3] = ambient_intensity;
	uniforms.albedo[3] = 1.0f;
	uniforms.specular[3] = material_shininess;
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
	// предыдущий кадр, поэтому буфер можно безопасно перезаписывать.
	graphics::writeBuffer(uniform_buffer, &uniforms, sizeof(uniforms));

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

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline_layout,
	                        0, 1, &vk_descriptor_set, 0, nullptr);

	const VkDeviceSize vertex_buffer_offset = 0;
	vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer.handle, &vertex_buffer_offset);

	vkCmdDraw(cmd, icosahedron_vertex_count, 1, 0, 0);

	// 5. Finish
	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

} // namespace application
