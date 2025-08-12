#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <cstdint>

namespace engine {

struct VulkanCommandsCreateInfo {
  VkDevice device = VK_NULL_HANDLE;
  uint32_t graphics_family = ~0u;
  uint32_t image_count = 0;
};

using RecordFunc = void(*)(VkCommandBuffer, VkImage, VkImageView, VkFormat, VkExtent2D, void*);

class VulkanCommands {
public:
  explicit VulkanCommands(const VulkanCommandsCreateInfo& ci);
  ~VulkanCommands();

  VulkanCommands(const VulkanCommands&) = delete;
  VulkanCommands& operator=(const VulkanCommands&) = delete;

  uint32_t acquire_record_present(
      VkSwapchainKHR swapchain,
      VkImage* swapchain_images,
      VkImageView* swapchain_views,
      VkFormat swapchain_format,
      VkExtent2D extent,
      VkQueue graphics_queue,
      VkQueue present_queue,
      RecordFunc recorder,
      void* user);

private:
  VkDevice dev_ = VK_NULL_HANDLE;
  uint32_t gfx_family_ = ~0u;

  struct Frame {
    VkSemaphore image_available = VK_NULL_HANDLE;
    VkSemaphore render_finished = VK_NULL_HANDLE;
    VkFence     in_flight = VK_NULL_HANDLE;
    VkCommandPool cmd_pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    bool first_use = true;
  };
  std::vector<Frame> frames_;
  uint32_t frame_cursor_ = 0;
};

} // namespace engine
