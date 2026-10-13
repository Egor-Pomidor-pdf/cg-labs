#include "graphics.hpp"

#include <cstring>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>

#include "graphics_internal.hpp"

namespace graphics {

bool createHostVisibleBuffer(size_t size, VkBufferUsageFlags usage, Buffer& out_buffer) {
	auto& context = internal::context;

	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	const VmaAllocationCreateInfo allocation_info = {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
		         VMA_ALLOCATION_CREATE_MAPPED_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	VmaAllocationInfo allocation = {};
	if (vmaCreateBuffer(context.allocator, &buffer_info, &allocation_info,
	                    &out_buffer.handle, &out_buffer.allocation, &allocation) != VK_SUCCESS) {
		std::cerr << "Failed to allocate and create Vulkan buffer\n";
		return false;
	}

	out_buffer.mapped = allocation.pMappedData;
	out_buffer.size = size;

	return true;
}

void writeBuffer(const Buffer& buffer, const void* data, size_t size) {
	if (buffer.mapped == nullptr || size > buffer.size) {
		return;
	}

	std::memcpy(buffer.mapped, data, size);

	// Makes the write visible to the GPU if the memory is not host coherent.
	vmaFlushAllocation(internal::context.allocator, buffer.allocation, 0, size);
}

void destroyBuffer(Buffer& buffer) {
	if (buffer.handle == VK_NULL_HANDLE) {
		return;
	}

	vmaDestroyBuffer(internal::context.allocator, buffer.handle, buffer.allocation);

	buffer.handle = VK_NULL_HANDLE;
	buffer.allocation = nullptr;
	buffer.mapped = nullptr;
	buffer.size = 0;
}

VkShaderModule loadShaderModule(const char* path) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file.is_open()) {
		std::cerr << "Failed to open shader file " << path
		          << " (is the working directory set to the project root?)\n";
		return VK_NULL_HANDLE;
	}

	const std::streamsize size = file.tellg();
	if (size <= 0 || (size % 4) != 0) {
		std::cerr << "Shader file " << path << " has an invalid size for SPIR-V\n";
		return VK_NULL_HANDLE;
	}

	std::vector<uint32_t> code(static_cast<size_t>(size) / sizeof(uint32_t));
	file.seekg(0);
	file.read(reinterpret_cast<char*>(code.data()), size);
	file.close();

	const VkShaderModuleCreateInfo shader_module = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = static_cast<size_t>(size),
		.pCode = code.data(),
	};

	VkShaderModule result = VK_NULL_HANDLE;
	if (vkCreateShaderModule(internal::context.device, &shader_module, nullptr,
	                         &result) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan shader module from " << path << '\n';
		return VK_NULL_HANDLE;
	}

	return result;
}

} // namespace graphics
