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

// Данные, которые шейдер получает напрямую из командного буфера.
struct PushConstants {
	Mat4 mvp;
	float tint[4]; // цвет из интерфейса, на него домножается цвет вершины
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

// Цвет фигуры из интерфейса. На него умножается процедурный цвет вершины.
float object_tint[3] = {1.0f, 1.0f, 1.0f};

double previous_time = 0.0;

// Матрица, посчитанная в update() и использованная в render().
Mat4 mvp = linalg::identity();

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

bool createPipeline() {
	auto& context = graphics::internal::context;

	VkShaderModule vertex_shader = graphics::loadShaderModule("shaders/object.vert.spv");
	VkShaderModule fragment_shader = graphics::loadShaderModule("shaders/object.frag.spv");
	if (vertex_shader == VK_NULL_HANDLE || fragment_shader == VK_NULL_HANDLE) {
		vkDestroyShaderModule(context.device, vertex_shader, nullptr);
		vkDestroyShaderModule(context.device, fragment_shader, nullptr);
		return false;
	}

	// Матрица MVP и цвет передаются через push-константы: 80 байт из
	// гарантированных спецификацией 128, так что набор дескрипторов не нужен.
	const VkPushConstantRange push_constant_range = {
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		.offset = 0,
		.size = sizeof(PushConstants),
	};

	const VkPipelineLayoutCreateInfo pipeline_layout = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &push_constant_range,
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
Mat4 modelMatrix() {
	Vec3 position = object_position;
	float extra_yaw = auto_rotate_angle;

	if (animation_playing || animation_time != 0.0f) {
		const Vec3 point = trajectoryPoint(animation_time);
		position = position + point;

		if (trajectory_turn) {
			// Фигура доворачивается по направлению движения: направление --
			// это разность двух близких точек траектории.
			const Vec3 ahead = trajectoryPoint(animation_time + 0.05f);
			const Vec3 velocity = ahead - point;
			extra_yaw += linalg::degrees(std::atan2(velocity.x, velocity.z));
		}
	}

	const Mat4 rotation = linalg::rotationZ(linalg::radians(object_rotation.z)) *
	                      linalg::rotationY(linalg::radians(object_rotation.y + extra_yaw)) *
	                      linalg::rotationX(linalg::radians(object_rotation.x));

	return linalg::translation(position) * rotation * linalg::scaling(object_scale);
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

	ImGui::SeparatorText("Colour");
	ImGui::ColorEdit3("tint", object_tint);
	ImGui::TextWrapped("Vertex colours are generated from vertex positions and "
	                   "multiplied by this colour in the shader.");

	ImGui::Separator();
	if (ImGui::Button("Reset")) {
		object_position = {0.0f, 0.0f, 0.0f};
		object_rotation = {0.0f, 0.0f, 0.0f};
		object_scale = {1.0f, 1.0f, 1.0f};
		auto_rotate_angle = 0.0f;
		animation_time = 0.0f;
		object_tint[0] = object_tint[1] = object_tint[2] = 1.0f;
	}

	ImGui::End();
}

} // namespace

bool initialize() {
	if (!createGeometry()) {
		return false;
	}

	return createPipeline();
}

void shutdown() {
	auto& context = graphics::internal::context;
	vkQueueWaitIdle(context.graphics_queue);

	vkDestroyPipeline(context.device, vk_pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, vk_pipeline_layout, nullptr);

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
	const Mat4 view = linalg::lookAt({0.0f, 0.0f, camera_distance},
	                                 {0.0f, 0.0f, 0.0f},
	                                 {0.0f, 1.0f, 0.0f});

	mvp = projectionMatrix() * view * modelMatrix();
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

	const PushConstants push_constants = {
		.mvp = mvp,
		.tint = {object_tint[0], object_tint[1], object_tint[2], 1.0f},
	};
	vkCmdPushConstants(cmd, vk_pipeline_layout,
	                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
	                   sizeof(push_constants), &push_constants);

	const VkDeviceSize vertex_buffer_offset = 0;
	vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer.handle, &vertex_buffer_offset);

	vkCmdDraw(cmd, icosahedron_vertex_count, 1, 0, 0);

	// 5. Finish
	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

} // namespace application
