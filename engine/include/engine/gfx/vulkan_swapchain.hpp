#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <cstdint>

namespace engine {

struct VulkanSwapchainCreateInfo {
  VkPhysicalDevice physical = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  uint32_t desired_width = 0;
  uint32_t desired_height = 0;
};

class VulkanSwapchain {
public:
  explicit VulkanSwapchain(const VulkanSwapchainCreateInfo& ci);
  ~VulkanSwapchain();

  VulkanSwapchain(const VulkanSwapchain&) = delete;
  VulkanSwapchain& operator=(const VulkanSwapchain&) = delete;

  VkSwapchainKHR vk() const { return swapchain_; }
  VkFormat image_format() const { return format_; }
  VkExtent2D extent() const { return extent_; }
  const std::vector<VkImageView>& image_views() const { return views_; }
  const std::vector<VkImage>& images() const { return images_; }  // <-- added

private:
  void create(const VulkanSwapchainCreateInfo& ci);
  void destroy();

  VkPhysicalDevice phys_ = VK_NULL_HANDLE;
  VkDevice dev_ = VK_NULL_HANDLE;
  VkSurfaceKHR surf_ = VK_NULL_HANDLE;

  VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
  VkFormat format_ = VK_FORMAT_UNDEFINED;
  VkColorSpaceKHR color_space_ = VK_COLORSPACE_SRGB_NONLINEAR_KHR;
  VkExtent2D extent_{};
  std::vector<VkImage> images_;
  std::vector<VkImageView> views_;
};

} // namespace engine
