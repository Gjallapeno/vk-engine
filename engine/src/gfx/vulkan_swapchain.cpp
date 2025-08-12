#include <engine/gfx/vulkan_swapchain.hpp>
#include <engine/vk_checks.hpp>
#include <spdlog/spdlog.h>
#include <algorithm>

namespace engine {

static VkSurfaceFormatKHR choose_format(const std::vector<VkSurfaceFormatKHR>& formats) {
  // Prefer SRGB 8-bit BGRA
  for (auto f : formats) {
    if (f.format == VK_FORMAT_B8G8R8A8_SRGB && f.colorSpace == VK_COLORSPACE_SRGB_NONLINEAR_KHR)
      return f;
  }
  // Fallback to first
  return formats.front();
}

static VkPresentModeKHR choose_present_mode(const std::vector<VkPresentModeKHR>& modes) {
  // Prefer MAILBOX, then FIFO (always available)
  for (auto m : modes) if (m == VK_PRESENT_MODE_MAILBOX_KHR) return m;
  return VK_PRESENT_MODE_FIFO_KHR;
}

static VkExtent2D choose_extent(const VkSurfaceCapabilitiesKHR& caps, uint32_t w, uint32_t h) {
  if (caps.currentExtent.width != 0xFFFFFFFF) {
    return caps.currentExtent;
  }
  VkExtent2D e{ std::clamp(w, caps.minImageExtent.width,  caps.maxImageExtent.width),
                std::clamp(h, caps.minImageExtent.height, caps.maxImageExtent.height) };
  return e;
}

VulkanSwapchain::VulkanSwapchain(const VulkanSwapchainCreateInfo& ci)
  : phys_(ci.physical), dev_(ci.device), surf_(ci.surface) {
  create(ci);
}

VulkanSwapchain::~VulkanSwapchain() { destroy(); }

void VulkanSwapchain::destroy() {
  for (auto v : views_) if (v) vkDestroyImageView(dev_, v, nullptr);
  views_.clear();
  images_.clear();
  if (swapchain_) {
    vkDestroySwapchainKHR(dev_, swapchain_, nullptr);
    swapchain_ = VK_NULL_HANDLE;
  }
}

void VulkanSwapchain::create(const VulkanSwapchainCreateInfo& ci) {
  // Query capabilities
  VkSurfaceCapabilitiesKHR caps{};
  VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(phys_, surf_, &caps));

  uint32_t fmtCount = 0, pmCount = 0;
  VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(phys_, surf_, &fmtCount, nullptr));
  std::vector<VkSurfaceFormatKHR> formats(fmtCount);
  VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(phys_, surf_, &fmtCount, formats.data()));

  VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(phys_, surf_, &pmCount, nullptr));
  std::vector<VkPresentModeKHR> modes(pmCount);
  VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(phys_, surf_, &pmCount, modes.data()));

  spdlog::info("[vk] Surface caps: minImages={} maxImages={} currentExtent={}x{}",
               caps.minImageCount, caps.maxImageCount, caps.currentExtent.width, caps.currentExtent.height);
  spdlog::info("[vk] Surface formats: {} | present modes: {}", fmtCount, pmCount);

  auto chosenFmt = choose_format(formats);
  auto chosenPM  = choose_present_mode(modes);
  auto chosenExt = choose_extent(caps, ci.desired_width, ci.desired_height);

  uint32_t desiredImages = std::max(caps.minImageCount + 1, 3u); // try triple buffer
  if (caps.maxImageCount > 0 && desiredImages > caps.maxImageCount) desiredImages = caps.maxImageCount;

  VkSwapchainCreateInfoKHR sci{};
  sci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  sci.surface = surf_;
  sci.minImageCount = desiredImages;
  sci.imageFormat = chosenFmt.format;
  sci.imageColorSpace = chosenFmt.colorSpace;
  sci.imageExtent = chosenExt;
  sci.imageArrayLayers = 1;
  sci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE; // graphics==present in our selection; update if split
  sci.preTransform = caps.currentTransform;
  sci.compositeAlpha = (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR)
                         ? VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR
                         : VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
  sci.presentMode = chosenPM;
  sci.clipped = VK_TRUE;

  VK_CHECK(vkCreateSwapchainKHR(dev_, &sci, nullptr, &swapchain_));

  uint32_t imgCount = 0;
  VK_CHECK(vkGetSwapchainImagesKHR(dev_, swapchain_, &imgCount, nullptr));
  images_.resize(imgCount);
  VK_CHECK(vkGetSwapchainImagesKHR(dev_, swapchain_, &imgCount, images_.data()));

  views_.resize(imgCount);
  for (uint32_t i = 0; i < imgCount; ++i) {
    VkImageViewCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image = images_[i];
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = chosenFmt.format;
    vi.components = { VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                      VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY };
    vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vi.subresourceRange.baseMipLevel = 0;
    vi.subresourceRange.levelCount = 1;
    vi.subresourceRange.baseArrayLayer = 0;
    vi.subresourceRange.layerCount = 1;
    VK_CHECK(vkCreateImageView(dev_, &vi, nullptr, &views_[i]));
  }

  format_ = chosenFmt.format;
  color_space_ = chosenFmt.colorSpace;
  extent_ = chosenExt;

  spdlog::info("[vk] Swapchain: {} images | {}x{} | fmt={} | present mode={}",
               imgCount, extent_.width, extent_.height,
               static_cast<int>(format_), static_cast<int>(chosenPM));
}

} // namespace engine
