// --- VMA implementation must come first in exactly one TU ---
#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vk_mem_alloc.h>
// ------------------------------------------------------------

#include <cstring>
#include <engine/gfx/memory.hpp>
#include <engine/vk_checks.hpp>

namespace engine {

namespace {

struct TransferContext {
  VkDevice device = VK_NULL_HANDLE;
  VkCommandPool pool = VK_NULL_HANDLE;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
};

TransferContext g_transfer;

void ensure_transfer(VkDevice device, uint32_t queue_family) {
  if (g_transfer.pool)
    return;

  g_transfer.device = device;

  VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pci.queueFamilyIndex = queue_family;
  pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  VK_CHECK(vkCreateCommandPool(device, &pci, nullptr, &g_transfer.pool));

  VkCommandBufferAllocateInfo ai{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ai.commandPool = g_transfer.pool;
  ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ai.commandBufferCount = 1;
  VK_CHECK(vkAllocateCommandBuffers(device, &ai, &g_transfer.cmd));

  VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VK_CHECK(vkCreateFence(device, &fi, nullptr, &g_transfer.fence));
}

} // namespace

// ---------- allocator ----------
void GpuAllocator::init(VkInstance instance, VkPhysicalDevice physical,
                        VkDevice device) {
  VmaVulkanFunctions f{};
  f.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
  f.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

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
Buffer create_buffer(VmaAllocator alloc, VkDeviceSize size,
                     VkBufferUsageFlags usage) {
  Buffer b{};
  b.size = size;

  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = size;
  bi.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT; // ensure copy capability
  bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo aci{};
  aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

  VK_CHECK(
      vmaCreateBuffer(alloc, &bi, &aci, &b.buffer, &b.allocation, nullptr));
  return b;
}

void destroy_buffer(VmaAllocator alloc, Buffer &buf) {
  if (buf.buffer) {
    vmaDestroyBuffer(alloc, buf.buffer, buf.allocation);
    buf = {};
  }
}

void upload_buffer(VmaAllocator alloc, VkDevice device, uint32_t queue_family,
                   VkQueue queue, const Buffer &dst, const void *data,
                   size_t bytes) {
  // staging buffer
  VkBuffer stagingBuf = VK_NULL_HANDLE;
  VmaAllocation stagingAlloc = nullptr;
  {
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = bytes;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo info{};
    VK_CHECK(
        vmaCreateBuffer(alloc, &bi, &aci, &stagingBuf, &stagingAlloc, &info));
    std::memcpy(info.pMappedData, data, bytes);
  }

  // command buffer
  ensure_transfer(device, queue_family);
  VK_CHECK(vkResetCommandPool(device, g_transfer.pool, 0));

  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VK_CHECK(vkBeginCommandBuffer(g_transfer.cmd, &bi));

  VkBufferCopy region{};
  region.size = bytes;
  vkCmdCopyBuffer(g_transfer.cmd, stagingBuf, dst.buffer, 1, &region);

  VK_CHECK(vkEndCommandBuffer(g_transfer.cmd));

  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &g_transfer.cmd;
  VK_CHECK(vkResetFences(device, 1, &g_transfer.fence));
  VK_CHECK(vkQueueSubmit(queue, 1, &si, g_transfer.fence));
  VK_CHECK(vkWaitForFences(device, 1, &g_transfer.fence, VK_TRUE, UINT64_MAX));

  vmaDestroyBuffer(alloc, stagingBuf, stagingAlloc);
}

// ---------- images ----------
static VkImageSubresourceRange full_color(uint32_t levels) {
  VkImageSubresourceRange r{};
  r.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  r.baseMipLevel = 0;
  r.levelCount = levels;
  r.baseArrayLayer = 0;
  r.layerCount = 1;
  return r;
}

Image2D create_image2d(VmaAllocator alloc, uint32_t w, uint32_t h,
                       VkFormat format, VkImageUsageFlags usage) {
  Image2D img{};
  img.width = w;
  img.height = h;
  img.mip_levels = 1;
  uint32_t max_dim = (w > h) ? w : h;
  while (max_dim > 1) {
    max_dim >>= 1;
    img.mip_levels++;
  }

  VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ici.imageType = VK_IMAGE_TYPE_2D;
  ici.format = format;
  ici.extent = {w, h, 1};
  ici.mipLevels = img.mip_levels;
  ici.arrayLayers = 1;
  ici.samples = VK_SAMPLE_COUNT_1_BIT;
  ici.tiling = VK_IMAGE_TILING_OPTIMAL;
  ici.usage =
      usage | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  VmaAllocationCreateInfo aci{};
  aci.usage = VMA_MEMORY_USAGE_AUTO;

  VK_CHECK(
      vmaCreateImage(alloc, &ici, &aci, &img.image, &img.allocation, nullptr));
  return img;
}

void destroy_image2d(VmaAllocator alloc, Image2D &img) {
  if (img.image) {
    vmaDestroyImage(alloc, img.image, img.allocation);
    img = {};
  }
}

void upload_image2d(VmaAllocator alloc, VkDevice device, uint32_t queue_family,
                    VkQueue queue, const void *src_rgba8, size_t src_bytes,
                    const Image2D &dst) {
  // staging
  VkBuffer stagingBuf = VK_NULL_HANDLE;
  VmaAllocation stagingAlloc = nullptr;
  {
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = src_bytes;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo info{};
    VK_CHECK(
        vmaCreateBuffer(alloc, &bi, &aci, &stagingBuf, &stagingAlloc, &info));
    std::memcpy(info.pMappedData, src_rgba8, src_bytes);
  }

  // command buffer
  ensure_transfer(device, queue_family);
  VK_CHECK(vkResetCommandPool(device, g_transfer.pool, 0));

  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VK_CHECK(vkBeginCommandBuffer(g_transfer.cmd, &bi));

  // layout transitions + copy
  VkImageMemoryBarrier to_dst{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  to_dst.srcAccessMask = 0;
  to_dst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  to_dst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  to_dst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  to_dst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_dst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_dst.image = dst.image;
  to_dst.subresourceRange = full_color(dst.mip_levels);

  vkCmdPipelineBarrier(g_transfer.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &to_dst);

  VkBufferImageCopy region{};
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.mipLevel = 0;
  region.imageSubresource.layerCount = 1;
  region.imageExtent = {dst.width, dst.height, 1};
  vkCmdCopyBufferToImage(g_transfer.cmd, stagingBuf, dst.image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

  uint32_t mip_w = dst.width;
  uint32_t mip_h = dst.height;
  for (uint32_t i = 1; i < dst.mip_levels; ++i) {
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = dst.image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = i - 1;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(g_transfer.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &barrier);

    VkImageBlit blit{};
    blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit.srcSubresource.mipLevel = i - 1;
    blit.srcSubresource.baseArrayLayer = 0;
    blit.srcSubresource.layerCount = 1;
    blit.srcOffsets[0] = {0, 0, 0};
    blit.srcOffsets[1] = {static_cast<int32_t>(mip_w), static_cast<int32_t>(mip_h), 1};
    mip_w = mip_w > 1 ? mip_w / 2 : 1;
    mip_h = mip_h > 1 ? mip_h / 2 : 1;
    blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit.dstSubresource.mipLevel = i;
    blit.dstSubresource.baseArrayLayer = 0;
    blit.dstSubresource.layerCount = 1;
    blit.dstOffsets[0] = {0, 0, 0};
    blit.dstOffsets[1] = {static_cast<int32_t>(mip_w), static_cast<int32_t>(mip_h), 1};
    vkCmdBlitImage(g_transfer.cmd, dst.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   dst.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                   VK_FILTER_LINEAR);

    barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vkCmdPipelineBarrier(g_transfer.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr,
                         0, nullptr, 1, &barrier);
  }

  VkImageMemoryBarrier last{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  last.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  last.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  last.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  last.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  last.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  last.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  last.image = dst.image;
  last.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  last.subresourceRange.baseMipLevel = dst.mip_levels - 1;
  last.subresourceRange.levelCount = 1;
  last.subresourceRange.baseArrayLayer = 0;
  last.subresourceRange.layerCount = 1;
  vkCmdPipelineBarrier(g_transfer.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &last);

  VK_CHECK(vkEndCommandBuffer(g_transfer.cmd));

  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &g_transfer.cmd;
  VK_CHECK(vkResetFences(device, 1, &g_transfer.fence));
  VK_CHECK(vkQueueSubmit(queue, 1, &si, g_transfer.fence));
    VK_CHECK(vkWaitForFences(device, 1, &g_transfer.fence, VK_TRUE, UINT64_MAX));

    vmaDestroyBuffer(alloc, stagingBuf, stagingAlloc);
  }

void destroy_transfer_context() {
  if (!g_transfer.pool && !g_transfer.fence)
    return;
  if (g_transfer.cmd)
    vkFreeCommandBuffers(g_transfer.device, g_transfer.pool, 1, &g_transfer.cmd);
  if (g_transfer.pool)
    vkDestroyCommandPool(g_transfer.device, g_transfer.pool, nullptr);
  if (g_transfer.fence)
    vkDestroyFence(g_transfer.device, g_transfer.fence, nullptr);
  g_transfer = {};
}

Image3D create_image3d(VmaAllocator alloc, uint32_t w, uint32_t h, uint32_t d,
                       VkFormat format, VkImageUsageFlags usage) {
  Image3D img{};
  img.width = w; img.height = h; img.depth = d;
  VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ci.imageType = VK_IMAGE_TYPE_3D;
  ci.format = format;
  ci.extent = {w, h, d};
  ci.mipLevels = 1;
  ci.arrayLayers = 1;
  ci.samples = VK_SAMPLE_COUNT_1_BIT;
  ci.tiling = VK_IMAGE_TILING_OPTIMAL;
  ci.usage = usage | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  VmaAllocationCreateInfo aci{};
  aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
  VK_CHECK(vmaCreateImage(alloc, &ci, &aci, &img.image, &img.allocation, nullptr));
  return img;
}

void destroy_image3d(VmaAllocator alloc, Image3D &img) {
  if (img.image) {
    vmaDestroyImage(alloc, img.image, img.allocation);
    img = {};
  }
}

void upload_image3d(VmaAllocator alloc, VkDevice device, uint32_t queue_family,
                    VkQueue queue, const void *src, size_t src_bytes,
                    const Image3D &dst) {
  VkBuffer stagingBuf = VK_NULL_HANDLE; VmaAllocation stagingAlloc = nullptr; {
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = src_bytes; bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VmaAllocationCreateInfo aci{}; aci.usage = VMA_MEMORY_USAGE_AUTO;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo info{};
    VK_CHECK(vmaCreateBuffer(alloc, &bi, &aci, &stagingBuf, &stagingAlloc, &info));
    std::memcpy(info.pMappedData, src, src_bytes);
  }

  ensure_transfer(device, queue_family);
  VK_CHECK(vkResetCommandPool(device, g_transfer.pool, 0));

  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VK_CHECK(vkBeginCommandBuffer(g_transfer.cmd, &bi));

  VkImageMemoryBarrier to_dst{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  to_dst.srcAccessMask = 0;
  to_dst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  to_dst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  to_dst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  to_dst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_dst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_dst.image = dst.image;
  to_dst.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  to_dst.subresourceRange.levelCount = 1;
  to_dst.subresourceRange.layerCount = 1;
  vkCmdPipelineBarrier(g_transfer.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &to_dst);

  VkBufferImageCopy region{};
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.mipLevel = 0;
  region.imageSubresource.baseArrayLayer = 0;
  region.imageSubresource.layerCount = 1;
  region.imageExtent = {dst.width, dst.height, dst.depth};
  vkCmdCopyBufferToImage(g_transfer.cmd, stagingBuf, dst.image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

  VkImageMemoryBarrier to_read{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  to_read.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  to_read.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  to_read.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  to_read.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  to_read.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_read.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_read.image = dst.image;
  to_read.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  to_read.subresourceRange.levelCount = 1;
  to_read.subresourceRange.layerCount = 1;
  vkCmdPipelineBarrier(g_transfer.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &to_read);

  VK_CHECK(vkEndCommandBuffer(g_transfer.cmd));

  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1; si.pCommandBuffers = &g_transfer.cmd;
  VK_CHECK(vkResetFences(device, 1, &g_transfer.fence));
  VK_CHECK(vkQueueSubmit(queue, 1, &si, g_transfer.fence));
  VK_CHECK(vkWaitForFences(device, 1, &g_transfer.fence, VK_TRUE, UINT64_MAX));

  vmaDestroyBuffer(alloc, stagingBuf, stagingAlloc);
}

} // namespace engine
