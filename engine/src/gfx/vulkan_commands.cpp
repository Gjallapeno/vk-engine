#include <engine/gfx/vulkan_commands.hpp>
#include <engine/vk_checks.hpp>
#include <spdlog/spdlog.h>
#include <algorithm>

namespace engine {

static VkImageSubresourceRange full_color_range() {
  VkImageSubresourceRange r{};
  r.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  r.baseMipLevel = 0;  r.levelCount = 1;
  r.baseArrayLayer = 0; r.layerCount = 1;
  return r;
}

VulkanCommands::VulkanCommands(const VulkanCommandsCreateInfo& ci)
  : dev_(ci.device), gfx_family_(ci.graphics_family)
{
  frames_.resize(std::max<uint32_t>(ci.image_count, 1u));

  for (auto& f : frames_) {
    VkSemaphoreTypeCreateInfo tci{ VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO };
    tci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    tci.initialValue = 0;
    VkSemaphoreCreateInfo sci{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    sci.pNext = &tci;
    VK_CHECK(vkCreateSemaphore(dev_, &sci, nullptr, &f.timeline));

    VkCommandPoolCreateInfo pci{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    pci.queueFamilyIndex = gfx_family_;
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(dev_, &pci, nullptr, &f.cmd_pool));

    VkCommandBufferAllocateInfo ai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    ai.commandPool = f.cmd_pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(dev_, &ai, &f.cmd));
  }
}

VulkanCommands::~VulkanCommands() {
  if (!dev_) return;
  // Ensure nothing is in flight before destroying sync/CMDBs
  vkDeviceWaitIdle(dev_);
  for (auto& f : frames_) {
    if (f.cmd)            vkFreeCommandBuffers(dev_, f.cmd_pool, 1, &f.cmd);
    if (f.cmd_pool)       vkDestroyCommandPool(dev_, f.cmd_pool, nullptr);
    if (f.timeline)       vkDestroySemaphore(dev_, f.timeline, nullptr);
  }
}

uint32_t VulkanCommands::acquire_record_present(
    VkSwapchainKHR swapchain,
    VkImage*       swapchain_images,
    VkImageView*   swapchain_image_views,
    VkFormat       swapchain_format,
    VkExtent2D     extent,
    VkQueue        graphics_queue,
    VkQueue        present_queue,
    RecordFunc     recorder,
    void*          user)
{
  auto& f = frames_[frame_cursor_];

  // Wait for previous work on this frame to finish
  VkSemaphoreWaitInfo frame_wait{ VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO };
  frame_wait.semaphoreCount = 1;
  frame_wait.pSemaphores = &f.timeline;
  frame_wait.pValues = &f.value;
  VK_CHECK(vkWaitSemaphores(dev_, &frame_wait, UINT64_MAX));

  // Acquire image
  uint32_t image_index = 0;
  uint64_t acquire_value = f.value + 1;
  VkResult acq = vkAcquireNextImageKHR(dev_, swapchain, UINT64_MAX, f.timeline, VK_NULL_HANDLE, &image_index);
  if (acq == VK_ERROR_OUT_OF_DATE_KHR) {
    spdlog::warn("[vk] Acquire returned OUT_OF_DATE (resize will recreate).");
    return image_index;
  }
  VK_CHECK(acq);

  // Record commands
  VK_CHECK(vkResetCommandBuffer(f.cmd, 0));
  VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VK_CHECK(vkBeginCommandBuffer(f.cmd, &bi));

  // Transition: PRESENT -> COLOR_ATTACHMENT_OPTIMAL (or UNDEFINED on first use)
  VkImageMemoryBarrier2 to_color{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
  to_color.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
  to_color.srcAccessMask = VK_ACCESS_2_NONE;
  to_color.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
  to_color.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
  to_color.oldLayout = f.first_use ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
  to_color.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  to_color.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_color.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_color.image = swapchain_images[image_index];
  to_color.subresourceRange = full_color_range();

  VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
  dep.imageMemoryBarrierCount = 1;
  dep.pImageMemoryBarriers = &to_color;
  vkCmdPipelineBarrier2(f.cmd, &dep);

  // User recording (dynamic rendering / draw calls etc.)
  if (recorder) {
    recorder(f.cmd,
             swapchain_images[image_index],
             swapchain_image_views[image_index],
             swapchain_format,
             extent,
             user);
  }

  // Transition back to PRESENT
  VkImageMemoryBarrier2 to_present{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
  to_present.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
  to_present.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
  to_present.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
  to_present.dstAccessMask = VK_ACCESS_2_NONE;
  to_present.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  to_present.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
  to_present.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_present.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_present.image = swapchain_images[image_index];
  to_present.subresourceRange = full_color_range();

  dep.imageMemoryBarrierCount = 1;
  dep.pImageMemoryBarriers = &to_present;
  vkCmdPipelineBarrier2(f.cmd, &dep);

  VK_CHECK(vkEndCommandBuffer(f.cmd));
  f.first_use = false;

  // Submit
  uint64_t submit_signal_value = acquire_value + 1;
  VkTimelineSemaphoreSubmitInfo tsi{ VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO };
  tsi.waitSemaphoreValueCount = 1;
  tsi.pWaitSemaphoreValues = &acquire_value;
  tsi.signalSemaphoreValueCount = 1;
  tsi.pSignalSemaphoreValues = &submit_signal_value;

  VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
  si.pNext = &tsi;
  si.waitSemaphoreCount = 1;
  si.pWaitSemaphores = &f.timeline;
  si.pWaitDstStageMask = &wait_stage;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &f.cmd;
  si.signalSemaphoreCount = 1;
  si.pSignalSemaphores = &f.timeline;

  VK_CHECK(vkQueueSubmit(graphics_queue, 1, &si, VK_NULL_HANDLE));

  // Wait for rendering to complete before presenting
  VkSemaphoreWaitInfo present_wait{ VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO };
  present_wait.semaphoreCount = 1;
  present_wait.pSemaphores = &f.timeline;
  present_wait.pValues = &submit_signal_value;
  VK_CHECK(vkWaitSemaphores(dev_, &present_wait, UINT64_MAX));

  // Present
  VkPresentInfoKHR pi{ VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
  pi.waitSemaphoreCount = 0;
  pi.swapchainCount = 1;
  pi.pSwapchains = &swapchain;
  pi.pImageIndices = &image_index;

  VkResult pres = vkQueuePresentKHR(present_queue, &pi);
  if (pres == VK_ERROR_OUT_OF_DATE_KHR || pres == VK_SUBOPTIMAL_KHR) {
    spdlog::warn("[vk] Present returned {} (resize will recreate).",
                 pres == VK_ERROR_OUT_OF_DATE_KHR ? "OUT_OF_DATE" : "SUBOPTIMAL");
  } else {
    VK_CHECK(pres);
  }

  // Update timeline value and next frame
  f.value = submit_signal_value;
  frame_cursor_ = (frame_cursor_ + 1) % static_cast<uint32_t>(frames_.size());
  return image_index;
}

} // namespace engine
