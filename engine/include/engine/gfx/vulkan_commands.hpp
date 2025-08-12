#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <vector>

namespace engine {

// Creation info for the per-frame command system.
struct VulkanCommandsCreateInfo {
  VkDevice device = VK_NULL_HANDLE;
  uint32_t graphics_family = ~0u;
  uint32_t image_count = 0; // swapchain image count
};

// Recorder callback: record rendering commands for one swapchain image.
// Arguments: cmd, swapchain image, its view, image format, extent, user pointer.
using RecordFunc = void(*)(VkCommandBuffer, VkImage, VkImageView, VkFormat, VkExtent2D, void*);

// Manages one command buffer + sync objects per swapchain image.
class VulkanCommands {
public:
  explicit VulkanCommands(const VulkanCommandsCreateInfo& ci);
  ~VulkanCommands();

  VulkanCommands(const VulkanCommands&) = delete;
  VulkanCommands& operator=(const VulkanCommands&) = delete;

  // Acquire next image, build commands via 'recorder', submit, and present.
  // Returns acquired image index.
  uint32_t acquire_record_present(
      VkSwapchainKHR swapchain,
      VkImage*       swapchain_images,
      VkImageView*   swapchain_image_views,
      VkFormat       swapchain_format,
      VkExtent2D     extent,
      VkQueue        graphics_queue,
      VkQueue        present_queue,
      RecordFunc     recorder,
      void*          user);

private:
  VkDevice  dev_ = VK_NULL_HANDLE;
  uint32_t  gfx_family_ = ~0u;

  struct Frame {
    VkSemaphore     timeline    = VK_NULL_HANDLE;
    uint64_t        value       = 0;            // last signaled value
    VkCommandPool   cmd_pool    = VK_NULL_HANDLE;
    VkCommandBuffer cmd         = VK_NULL_HANDLE;
    bool            first_use   = true;
  };
  std::vector<Frame> frames_;
  uint32_t frame_cursor_ = 0;
};

} // namespace engine
