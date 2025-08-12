#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <optional>
#include <cstdint>

namespace engine {

struct VulkanDeviceCreateInfo {
  VkInstance instance = VK_NULL_HANDLE;
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  bool enable_validation = true;
};

class VulkanDevice {
public:
  explicit VulkanDevice(const VulkanDeviceCreateInfo& ci);
  ~VulkanDevice();

  VulkanDevice(const VulkanDevice&) = delete;
  VulkanDevice& operator=(const VulkanDevice&) = delete;

  VkPhysicalDevice physical() const { return phys_; }
  VkDevice device() const { return dev_; }
  uint32_t graphics_family() const { return q_gfx_index_; }
  uint32_t present_family()  const { return q_present_index_; }
  VkQueue  graphics_queue()  const { return q_gfx_; }
  VkQueue  present_queue()   const { return q_present_; }

private:
  void pick_physical(VkInstance instance, VkSurfaceKHR surface);
  void create_logical(bool enable_validation);

  VkInstance instance_ = VK_NULL_HANDLE;
  VkSurfaceKHR surface_ = VK_NULL_HANDLE;

  VkPhysicalDevice phys_ = VK_NULL_HANDLE;
  VkDevice dev_ = VK_NULL_HANDLE;

  uint32_t q_gfx_index_ = ~0u;
  uint32_t q_present_index_ = ~0u;
  VkQueue  q_gfx_ = VK_NULL_HANDLE;
  VkQueue  q_present_ = VK_NULL_HANDLE;
};

} // namespace engine
