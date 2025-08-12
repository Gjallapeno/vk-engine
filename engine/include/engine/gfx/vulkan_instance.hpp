#pragma once
#include <vector>
#include <string>
#include <vulkan/vulkan.h>

namespace engine {

struct VulkanInstanceCreateInfo {
  bool enable_validation = true;
  // extra instance extensions can be added later (e.g., GLFW surface ones)
  std::vector<const char*> extra_extensions;
};

class VulkanInstance {
public:
  explicit VulkanInstance(const VulkanInstanceCreateInfo& ci);
  ~VulkanInstance();

  VulkanInstance(const VulkanInstance&) = delete;
  VulkanInstance& operator=(const VulkanInstance&) = delete;

  VkInstance vk() const { return instance_; }

private:
  void create_instance(const VulkanInstanceCreateInfo& ci);
  void setup_debug_messenger(bool enable_validation);
  static VKAPI_ATTR VkBool32 VKAPI_CALL debug_cb(
      VkDebugUtilsMessageSeverityFlagBitsEXT severity,
      VkDebugUtilsMessageTypeFlagsEXT types,
      const VkDebugUtilsMessengerCallbackDataEXT* data,
      void* user_data);

private:
  VkInstance instance_ = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT dbg_messenger_ = VK_NULL_HANDLE;

  PFN_vkCreateDebugUtilsMessengerEXT  pfnCreateDbg_  = nullptr;
  PFN_vkDestroyDebugUtilsMessengerEXT pfnDestroyDbg_ = nullptr;
};

// Debug label helpers
extern PFN_vkCmdBeginDebugUtilsLabelEXT  pfnCmdBeginDebugUtilsLabelEXT;
extern PFN_vkCmdEndDebugUtilsLabelEXT    pfnCmdEndDebugUtilsLabelEXT;
extern PFN_vkCmdInsertDebugUtilsLabelEXT pfnCmdInsertDebugUtilsLabelEXT;

// Load debug utils label commands after instance/device creation
void load_debug_label_functions(VkInstance instance, VkDevice device);

} // namespace engine

#define vkCmdBeginDebugUtilsLabelEXT  engine::pfnCmdBeginDebugUtilsLabelEXT
#define vkCmdEndDebugUtilsLabelEXT    engine::pfnCmdEndDebugUtilsLabelEXT
#define vkCmdInsertDebugUtilsLabelEXT engine::pfnCmdInsertDebugUtilsLabelEXT
