#pragma once
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <cstdint>

namespace engine {

// Simple RAII wrapper for VMA
class GpuAllocator {
public:
  GpuAllocator() = default;
  ~GpuAllocator() = default;

  void init(VkInstance instance, VkPhysicalDevice physical, VkDevice device);
  void destroy();

  VmaAllocator raw() const { return alloc_; }

private:
  VmaAllocator alloc_ = nullptr;
};

// -------- Buffers --------
struct Buffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VmaAllocation allocation = nullptr;
  VkDeviceSize size = 0;
};

Buffer create_buffer(VmaAllocator alloc, VkDeviceSize size, VkBufferUsageFlags usage);
void   destroy_buffer(VmaAllocator alloc, Buffer& buf);
// One-time upload via a transient command pool on 'queue_family' using 'queue'.
void   upload_buffer(VmaAllocator alloc,
                     VkDevice device,
                     uint32_t queue_family,
                     VkQueue queue,
                     const Buffer& dst,
                     const void* data,
                     size_t bytes);

// -------- Images --------
struct Image2D {
  VkImage       image = VK_NULL_HANDLE;
  VmaAllocation allocation = nullptr;
  uint32_t      width = 0, height = 0;
};

Image2D create_image2d(VmaAllocator alloc,
                       uint32_t w, uint32_t h,
                       VkFormat format,
                       VkImageUsageFlags usage);

void destroy_image2d(VmaAllocator alloc, Image2D& img);

// One-time upload via a transient command pool on 'queue_family' using 'queue'.
// Transitions: UNDEFINED -> TRANSFER_DST, copy, TRANSFER_DST -> SHADER_READ_ONLY.
void upload_image2d(VmaAllocator alloc,
                    VkDevice device,
                    uint32_t queue_family,
                    VkQueue queue,
                    const void* src_rgba8,
                    size_t src_bytes,
                    const Image2D& dst);

} // namespace engine
