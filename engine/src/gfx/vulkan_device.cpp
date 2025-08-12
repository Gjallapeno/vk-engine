#include <engine/gfx/vulkan_device.hpp>
#include <engine/vk_checks.hpp>
#include <spdlog/spdlog.h>
#include <vector>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <cstdint>

namespace engine {

static bool supports_present(VkPhysicalDevice pd, uint32_t qf, VkSurfaceKHR surf) {
  VkBool32 supported = VK_FALSE;
  vkGetPhysicalDeviceSurfaceSupportKHR(pd, qf, surf, &supported);
  return supported == VK_TRUE;
}

static const char* kPipelineCacheFile = "pipeline_cache.bin";

VulkanDevice::VulkanDevice(const VulkanDeviceCreateInfo& ci)
  : instance_(ci.instance), surface_(ci.surface) {
  pick_physical(instance_, surface_);
  create_logical(ci.enable_validation);
  create_pipeline_cache();
}

VulkanDevice::~VulkanDevice() {
  if (cache_) {
    size_t size = 0;
    VK_CHECK(vkGetPipelineCacheData(dev_, cache_, &size, nullptr));
    if (size > 0) {
      std::vector<uint8_t> data(size);
      VK_CHECK(vkGetPipelineCacheData(dev_, cache_, &size, data.data()));
      std::ofstream f(kPipelineCacheFile, std::ios::binary | std::ios::trunc);
      if (f) f.write(reinterpret_cast<char*>(data.data()), size);
    }
    vkDestroyPipelineCache(dev_, cache_, nullptr);
  }
  if (dev_) vkDestroyDevice(dev_, nullptr);
}

void VulkanDevice::pick_physical(VkInstance instance, VkSurfaceKHR surface) {
  uint32_t count = 0;
  VK_CHECK(vkEnumeratePhysicalDevices(instance, &count, nullptr));
  std::vector<VkPhysicalDevice> pds(count);
  VK_CHECK(vkEnumeratePhysicalDevices(instance, &count, pds.data()));
  if (pds.empty()) {
    spdlog::error("[vk] No physical devices found.");
    std::abort();
  }

  struct Candidate {
    VkPhysicalDevice pd;
    int score;
    uint32_t gfx;
    uint32_t present;
  };
  std::vector<Candidate> cands;

  for (auto pd : pds) {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(pd, &props);

    uint32_t qcount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qcount, nullptr);
    std::vector<VkQueueFamilyProperties> qprops(qcount);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qcount, qprops.data());

    std::optional<uint32_t> gfx, present;
    for (uint32_t i = 0; i < qcount; ++i) {
      if (qprops[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) gfx = i;
      if (supports_present(pd, i, surface)) present = i;
    }

    int score = 0;
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) score += 1000;
    if (gfx.has_value() && present.has_value()) score += 100;
    score += static_cast<int>(props.limits.maxImageDimension2D / 1024);

    spdlog::info("[vk] GPU: {} | type={} | score={} | gfx={} present={}",
                 props.deviceName,
                 static_cast<int>(props.deviceType),    // <- cast fixes fmt error
                 score,
                 gfx.has_value() ? *gfx : 0xFFFFFFFFu,
                 present.has_value() ? *present : 0xFFFFFFFFu);

    if (gfx && present) {
      cands.push_back({pd, score, *gfx, *present});
    }
  }

  if (cands.empty()) {
    spdlog::error("[vk] No device with both graphics+present queues.");
    std::abort();
  }

  std::sort(cands.begin(), cands.end(), [](const Candidate& a, const Candidate& b){
    return a.score > b.score;
  });

  phys_ = cands.front().pd;
  q_gfx_index_ = cands.front().gfx;
  q_present_index_ = cands.front().present;

  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(phys_, &props);
  spdlog::info("[vk] Chosen GPU: {}", props.deviceName);
  spdlog::info("[vk] Queues: graphics={} present={}", q_gfx_index_, q_present_index_);
}

void VulkanDevice::create_logical(bool enable_validation) {
    (void)enable_validation;
  
    float prio = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> qinfos;
  
    std::vector<uint32_t> unique_indices;
    unique_indices.push_back(q_gfx_index_);
    if (q_present_index_ != q_gfx_index_) unique_indices.push_back(q_present_index_);
  
    for (auto idx : unique_indices) {
      VkDeviceQueueCreateInfo qi{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
      qi.queueFamilyIndex = idx;
      qi.queueCount = 1;
      qi.pQueuePriorities = &prio;
      qinfos.push_back(qi);
    }
  
    VkPhysicalDeviceFeatures feats{};
    VkPhysicalDeviceFeatures supported{};
    vkGetPhysicalDeviceFeatures(phys_, &supported);
    if (supported.samplerAnisotropy)
      feats.samplerAnisotropy = VK_TRUE;
    if (supported.fragmentStoresAndAtomics)
      feats.fragmentStoresAndAtomics = VK_TRUE;
  
    // Dynamic rendering feature
    VkPhysicalDeviceDynamicRenderingFeatures dyn{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES };
    dyn.dynamicRendering = VK_TRUE;

    const char* kExts[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

    VkDeviceCreateInfo di{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    di.pNext = &dyn;
    di.queueCreateInfoCount = static_cast<uint32_t>(qinfos.size());
    di.pQueueCreateInfos = qinfos.data();
    di.pEnabledFeatures = &feats;
    di.enabledExtensionCount = 1;
    di.ppEnabledExtensionNames = kExts;
  
    VK_CHECK(vkCreateDevice(phys_, &di, nullptr, &dev_));
  
    vkGetDeviceQueue(dev_, q_gfx_index_, 0, &q_gfx_);
    vkGetDeviceQueue(dev_, q_present_index_, 0, &q_present_);
    spdlog::info("[vk] Logical device created. gfxQ={} presentQ={}", q_gfx_index_, q_present_index_);
  }

void VulkanDevice::create_pipeline_cache() {
  std::vector<uint8_t> data;
  std::ifstream f(kPipelineCacheFile, std::ios::binary | std::ios::ate);
  if (f) {
    size_t sz = static_cast<size_t>(f.tellg());
    data.resize(sz);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(data.data()), sz);
  }
  VkPipelineCacheCreateInfo ci{ VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO };
  ci.initialDataSize = data.size();
  ci.pInitialData = data.data();
  VK_CHECK(vkCreatePipelineCache(dev_, &ci, nullptr, &cache_));
}
  
} // namespace engine
