#pragma once
#include <cstdint>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

namespace engine {

// Owns resources required for one-off transfer operations.
class TransferContext {
public:
  TransferContext(VkDevice device, uint32_t queue_family);
  ~TransferContext();

  VkDevice device = VK_NULL_HANDLE;
  VkCommandPool pool = VK_NULL_HANDLE;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
};

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

Buffer create_buffer(VmaAllocator alloc, VkDeviceSize size,
                     VkBufferUsageFlags usage);
// Host-visible buffer intended for readback/cpu access.
Buffer create_host_buffer(VmaAllocator alloc, VkDeviceSize size,
                          VkBufferUsageFlags usage);
void destroy_buffer(VmaAllocator alloc, Buffer &buf);
// Upload using the provided transfer context on 'queue'.
void upload_buffer(VmaAllocator alloc, TransferContext &ctx, VkQueue queue,
                   const Buffer &dst, const void *data, size_t bytes);

// -------- Images --------
struct Image2D {
  VkImage image = VK_NULL_HANDLE;
  VmaAllocation allocation = nullptr;
  uint32_t width = 0, height = 0;
  uint32_t mip_levels = 1;
};

Image2D create_image2d(VmaAllocator alloc, uint32_t w, uint32_t h,
                       VkFormat format, VkImageUsageFlags usage);

void destroy_image2d(VmaAllocator alloc, Image2D &img);

// Upload using the provided transfer context on 'queue'. Transitions:
// UNDEFINED -> TRANSFER_DST, copy, TRANSFER_DST -> SHADER_READ_ONLY.
void upload_image2d(VmaAllocator alloc, TransferContext &ctx, VkQueue queue,
                    const void *src_rgba8, size_t src_bytes,
                    const Image2D &dst);

struct Image3D {
  VkImage image = VK_NULL_HANDLE;
  VmaAllocation allocation = nullptr;
  uint32_t width = 0, height = 0, depth = 0;
};

Image3D create_image3d(VmaAllocator alloc, uint32_t w, uint32_t h, uint32_t d,
                       VkFormat format, VkImageUsageFlags usage);
void destroy_image3d(VmaAllocator alloc, Image3D &img);
void upload_image3d(VmaAllocator alloc, TransferContext &ctx, VkQueue queue,
                    const void *src, size_t src_bytes, const Image3D &dst);

} // namespace engine
