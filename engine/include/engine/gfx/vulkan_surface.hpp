#pragma once
#include <vulkan/vulkan.h>

namespace engine {

class VulkanSurface {
public:
  VulkanSurface(VkInstance instance, void* glfw_window);
  ~VulkanSurface();

  VulkanSurface(const VulkanSurface&) = delete;
  VulkanSurface& operator=(const VulkanSurface&) = delete;

  VkSurfaceKHR vk() const { return surf_; }

private:
  VkInstance instance_ = VK_NULL_HANDLE;
  VkSurfaceKHR surf_ = VK_NULL_HANDLE;
};

} // namespace engine
