#include "application.hpp"

#include <imgui.h>

namespace application
{

	bool initialize()
	{
		return true;
	}

	void shutdown()
	{
		auto &context = graphics::internal::context;
		vkQueueWaitIdle(context.graphics_queue);
	}

	void update([[maybe_unused]] double time)
	{
		ImGui::ShowDemoWindow();
	}

	void render(const graphics::internal::FrameData &fd)
	{
		auto &context = graphics::internal::context;
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

		// 4. Later your drawing goes here:
		//    vkCmdBindPipeline, vkCmdSetViewport/Scissor,
		//    vkCmdBindVertexBuffers, vkCmdBindIndexBuffer, vkCmdDrawIndexed ...

		// 5. Finish
		vkCmdEndRenderPass(cmd);
		vkEndCommandBuffer(cmd);
	}

} // namespace application