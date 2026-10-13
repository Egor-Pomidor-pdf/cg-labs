#pragma once

#include <cstddef>

#include <vulkan/vulkan_core.h>

#include <vk_mem_alloc.h>

// Small reusable wrappers over the repetitive Vulkan code (buffers, shader
// modules). Everything specific to a particular lab lives in application.cpp.

namespace graphics {

struct Buffer {
	VkBuffer handle = VK_NULL_HANDLE;
	VmaAllocation allocation = nullptr;
	void* mapped = nullptr;
	size_t size = 0;
};

// Allocates a buffer in memory that is visible both to the CPU and the GPU and
// keeps it permanently mapped. For the small amount of geometry used in this
// lab it is enough; a staging buffer + vkCmdCopyBuffer into device-local
// memory would be the faster option for big meshes.
bool createHostVisibleBuffer(size_t size, VkBufferUsageFlags usage, Buffer& out_buffer);

// Copies `size` bytes into an already created host visible buffer.
void writeBuffer(const Buffer& buffer, const void* data, size_t size);

void destroyBuffer(Buffer& buffer);

// Reads a compiled SPIR-V file (*.spv) and creates a shader module out of it.
// The path is relative to the working directory, which must be the project root.
VkShaderModule loadShaderModule(const char* path);

} // namespace graphics
