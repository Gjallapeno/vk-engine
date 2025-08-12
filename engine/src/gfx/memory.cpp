// --- VMA implementation must come first in exactly one TU ---
#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vk_mem_alloc.h>
// ------------------------------------------------------------

#include <engine/gfx/memory.hpp>
#include <engine/vk_checks.hpp>
#include <cstring>

namespace engine {

// ---------- allocator ----------
void GpuAllocator::init(VkInstance instance, VkPhysicalDevice physical, VkDevice device) {
  VmaVulkanFunctions f{};
  f.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
  f.vkGetDeviceProcAddr   = vkGetDeviceProcAddr;

  VmaAllocatorCreateInfo ci{};
  ci.instance = instance;
  ci.physicalDevice = physical;
  ci.device = device;
  ci.pVulkanFunctions = &f;

  VK_CHECK(vmaCreateAllocator(&ci, &alloc_));
}

void GpuAllocator::destroy() {
  if (alloc_) {
    vmaDestroyAllocator(alloc_);
    alloc_ = nullptr;
  }
}

// ---------- buffers ----------
Buffer create_buffer(VmaAllocator alloc, VkDeviceSize size, VkBufferUsageFlags usage) {
  Buffer b{};
  b.size = size;

  VkBufferCreateInfo bi{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
  bi.size = size;
  bi.usage = usage; // vertex/index will pass VK_BUFFER_USAGE_VERTEX_BUFFER_BIT / INDEX_BUFFER_BIT
  bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo aci{};
  aci.usage = VMA_MEMORY_USAGE_AUTO;

  // IMPORTANT: allow CPU mapping on these buffers to avoid VMA assert.
  // We'll move to a staging copy in a later step for perf.
  aci.flags =
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
      VMA_ALLOCATION_CREATE_MAPPED_BIT;

  VK_CHECK(vmaCreateBuffer(alloc, &bi, &aci, &b.buffer, &b.allocation, nullptr));
  return b;
}

void destroy_buffer(VmaAllocator alloc, Buffer& buf) {
  if (buf.buffer) {
    vmaDestroyBuffer(alloc, buf.buffer, buf.allocation);
    buf = {};
  }
}

void upload_buffer(VmaAllocator alloc, const Buffer& dst, const void* data, size_t bytes) {
  void* mapped = nullptr;
  VK_CHECK(vmaMapMemory(alloc, dst.allocation, &mapped));
  std::memcpy(mapped, data, bytes);
  vmaUnmapMemory(alloc, dst.allocation);
}

// ---------- images ----------
static VkImageSubresourceRange full_color() {
  VkImageSubresourceRange r{};
  r.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  r.baseMipLevel = 0; r.levelCount = 1;
  r.baseArrayLayer = 0; r.layerCount = 1;
  return r;
}

Image2D create_image2d(VmaAllocator alloc,
                       uint32_t w, uint32_t h,
                       VkFormat format,
                       VkImageUsageFlags usage)
{
  Image2D img{};
  img.width = w; img.height = h;

  VkImageCreateInfo ici{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
  ici.imageType = VK_IMAGE_TYPE_2D;
  ici.format = format;
  ici.extent = { w, h, 1 };
  ici.mipLevels = 1;
  ici.arrayLayers = 1;
  ici.samples = VK_SAMPLE_COUNT_1_BIT;
  ici.tiling = VK_IMAGE_TILING_OPTIMAL;
  ici.usage = usage | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  VmaAllocationCreateInfo aci{};
  aci.usage = VMA_MEMORY_USAGE_AUTO;

  VK_CHECK(vmaCreateImage(alloc, &ici, &aci, &img.image, &img.allocation, nullptr));
  return img;
}

void destroy_image2d(VmaAllocator alloc, Image2D& img) {
  if (img.image) {
    vmaDestroyImage(alloc, img.image, img.allocation);
    img = {};
  }
}

void upload_image2d(VmaAllocator alloc,
                    VkDevice device,
                    uint32_t queue_family,
                    VkQueue queue,
                    const void* src_rgba8,
                    size_t src_bytes,
                    const Image2D& dst)
{
  // staging
  VkBuffer stagingBuf = VK_NULL_HANDLE;
  VmaAllocation stagingAlloc = nullptr;
  {
    VkBufferCreateInfo bi{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bi.size  = src_bytes;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo info{};
    VK_CHECK(vmaCreateBuffer(alloc, &bi, &aci, &stagingBuf, &stagingAlloc, &info));
    std::memcpy(info.pMappedData, src_rgba8, src_bytes);
  }

  // transient cmd
  VkCommandPool pool = VK_NULL_HANDLE;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  {
    VkCommandPoolCreateInfo pci{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    pci.queueFamilyIndex = queue_family;
    pci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(device, &pci, nullptr, &pool));

    VkCommandBufferAllocateInfo ai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    ai.commandPool = pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(device, &ai, &cmd));

    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
  }

  // layout transitions + copy
  VkImageMemoryBarrier to_dst{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
  to_dst.srcAccessMask = 0;
  to_dst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  to_dst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  to_dst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  to_dst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_dst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_dst.image = dst.image;
  to_dst.subresourceRange = full_color();

  vkCmdPipelineBarrier(cmd,
    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
    0, 0, nullptr, 0, nullptr, 1, &to_dst);

  VkBufferImageCopy region{};
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.layerCount = 1;
  region.imageExtent = { dst.width, dst.height, 1 };
  vkCmdCopyBufferToImage(cmd, stagingBuf, dst.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

  VkImageMemoryBarrier to_read{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
  to_read.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  to_read.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  to_read.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  to_read.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  to_read.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_read.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_read.image = dst.image;
  to_read.subresourceRange = full_color();

  vkCmdPipelineBarrier(cmd,
    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
    0, 0, nullptr, 0, nullptr, 1, &to_read);

  VK_CHECK(vkEndCommandBuffer(cmd));

  VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  VK_CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
  VK_CHECK(vkQueueWaitIdle(queue));

  vkFreeCommandBuffers(device, pool, 1, &cmd);
  vkDestroyCommandPool(device, pool, nullptr);
  vmaDestroyBuffer(alloc, stagingBuf, stagingAlloc);
}

} // namespace engine
