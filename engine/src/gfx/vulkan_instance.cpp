#include <engine/gfx/vulkan_instance.hpp>
#include <engine/config.hpp>
#include <engine/vk_checks.hpp>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cstring>

namespace engine {

PFN_vkCmdBeginDebugUtilsLabelEXT  pfnCmdBeginDebugUtilsLabelEXT  = nullptr;
PFN_vkCmdEndDebugUtilsLabelEXT    pfnCmdEndDebugUtilsLabelEXT    = nullptr;
PFN_vkCmdInsertDebugUtilsLabelEXT pfnCmdInsertDebugUtilsLabelEXT = nullptr;

static bool has_layer(const std::vector<VkLayerProperties>& layers, const char* name) {
  for (auto& l : layers) if (std::strcmp(l.layerName, name) == 0) return true;
  return false;
}

static bool has_extension(const std::vector<VkExtensionProperties>& exts, const char* name) {
  for (auto& e : exts) if (std::strcmp(e.extensionName, name) == 0) return true;
  return false;
}

VulkanInstance::VulkanInstance(const VulkanInstanceCreateInfo& ci) {
  create_instance(ci);
  setup_debug_messenger(ci.enable_validation);
}

VulkanInstance::~VulkanInstance() {
  if (dbg_messenger_ && pfnDestroyDbg_) {
    pfnDestroyDbg_(instance_, dbg_messenger_, nullptr);
  }
  if (instance_) {
    vkDestroyInstance(instance_, nullptr);
  }
}

void VulkanInstance::create_instance(const VulkanInstanceCreateInfo& ci) {
  // Enumerate layers & extensions for logging and checks
  uint32_t layer_count = 0;
  vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
  std::vector<VkLayerProperties> layers(layer_count);
  if (layer_count) vkEnumerateInstanceLayerProperties(&layer_count, layers.data());

  uint32_t ext_count = 0;
  vkEnumerateInstanceExtensionProperties(nullptr, &ext_count, nullptr);
  std::vector<VkExtensionProperties> exts(ext_count);
  if (ext_count) vkEnumerateInstanceExtensionProperties(nullptr, &ext_count, exts.data());

  spdlog::info("[vk] Instance layers ({}):", layers.size());
  for (auto& l : layers) spdlog::trace("  - {} v{}.{}", l.layerName, VK_VERSION_MAJOR(l.specVersion), VK_VERSION_MINOR(l.specVersion));
  spdlog::info("[vk] Instance extensions ({}):", exts.size());
  for (auto& e : exts) spdlog::trace("  - {}", e.extensionName);

  // Build requested lists
  std::vector<const char*> enabled_layers;
  std::vector<const char*> enabled_exts;

  const char* kValidationLayer = "VK_LAYER_KHRONOS_validation";
  const char* kDebugUtilsExt  = VK_EXT_DEBUG_UTILS_EXTENSION_NAME; // "VK_EXT_debug_utils"

  if (ci.enable_validation && has_layer(layers, kValidationLayer)) {
    enabled_layers.push_back(kValidationLayer);
  } else if (ci.enable_validation) {
    spdlog::warn("[vk] validation requested but {} not present; continuing without it.", kValidationLayer);
  }

  if (has_extension(exts, kDebugUtilsExt)) {
    enabled_exts.push_back(kDebugUtilsExt);
  } else {
    spdlog::warn("[vk] {} not present; debug names/messages will be unavailable.", kDebugUtilsExt);
  }

  for (auto* x : ci.extra_extensions) enabled_exts.push_back(x);

  // App info
  VkApplicationInfo app{};
  app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app.pApplicationName = "vk_engine";
  app.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
  app.pEngineName = "vk_engine";
  app.engineVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
  app.apiVersion = VK_API_VERSION_1_3;

  VkInstanceCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  info.pApplicationInfo = &app;
  info.enabledLayerCount = static_cast<uint32_t>(enabled_layers.size());
  info.ppEnabledLayerNames = enabled_layers.empty() ? nullptr : enabled_layers.data();
  info.enabledExtensionCount = static_cast<uint32_t>(enabled_exts.size());
  info.ppEnabledExtensionNames = enabled_exts.empty() ? nullptr : enabled_exts.data();

  VK_CHECK(vkCreateInstance(&info, nullptr, &instance_));

  spdlog::info("[vk] Instance created. Layers: {} Exts: {}", enabled_layers.size(), enabled_exts.size());
  for (auto* l : enabled_layers) spdlog::info("  * layer: {}", l);
  for (auto* e : enabled_exts)   spdlog::info("  * ext:   {}", e);
}

void VulkanInstance::setup_debug_messenger(bool enable_validation) {
  pfnCreateDbg_  = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
      vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT"));
  pfnDestroyDbg_ = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
      vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));

  if (!pfnCreateDbg_ || !pfnDestroyDbg_) {
    spdlog::warn("[vk] Debug Utils function pointers not available.");
    return;
  }

  if (!enable_validation) return;

  VkDebugUtilsMessengerCreateInfoEXT ci{};
  ci.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  ci.messageSeverity =
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT;
  ci.messageType =
      VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  ci.pfnUserCallback = &VulkanInstance::debug_cb;
  ci.pUserData = nullptr;

  VK_CHECK(pfnCreateDbg_(instance_, &ci, nullptr, &dbg_messenger_));
  spdlog::info("[vk] Debug messenger active.");
}

// Static
VKAPI_ATTR VkBool32 VKAPI_CALL VulkanInstance::debug_cb(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void*) {

  const char* sev =
    (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) ? "ERROR" :
    (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) ? "WARN" :
    (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) ? "INFO" : "VERBOSE";

  // Include message id if available
  if (data && data->pMessageIdName) {
    spdlog::info("[vk][{}] {}: {}", sev, data->pMessageIdName, data->pMessage);
  } else if (data) {
    spdlog::info("[vk][{}] {}", sev, data->pMessage);
  } else {
    spdlog::info("[vk][{}] <no data>", sev);
  }
  return VK_FALSE; // do not abort
}

void load_debug_label_functions(VkInstance instance, VkDevice device) {
  pfnCmdBeginDebugUtilsLabelEXT =
      reinterpret_cast<PFN_vkCmdBeginDebugUtilsLabelEXT>(
          vkGetInstanceProcAddr(instance, "vkCmdBeginDebugUtilsLabelEXT"));
  pfnCmdEndDebugUtilsLabelEXT =
      reinterpret_cast<PFN_vkCmdEndDebugUtilsLabelEXT>(
          vkGetInstanceProcAddr(instance, "vkCmdEndDebugUtilsLabelEXT"));
  pfnCmdInsertDebugUtilsLabelEXT =
      reinterpret_cast<PFN_vkCmdInsertDebugUtilsLabelEXT>(
          vkGetInstanceProcAddr(instance, "vkCmdInsertDebugUtilsLabelEXT"));

  if (device) {
    pfnCmdBeginDebugUtilsLabelEXT =
        reinterpret_cast<PFN_vkCmdBeginDebugUtilsLabelEXT>(
            vkGetDeviceProcAddr(device, "vkCmdBeginDebugUtilsLabelEXT"));
    pfnCmdEndDebugUtilsLabelEXT =
        reinterpret_cast<PFN_vkCmdEndDebugUtilsLabelEXT>(
            vkGetDeviceProcAddr(device, "vkCmdEndDebugUtilsLabelEXT"));
    pfnCmdInsertDebugUtilsLabelEXT =
        reinterpret_cast<PFN_vkCmdInsertDebugUtilsLabelEXT>(
            vkGetDeviceProcAddr(device, "vkCmdInsertDebugUtilsLabelEXT"));
  }
}

} // namespace engine
