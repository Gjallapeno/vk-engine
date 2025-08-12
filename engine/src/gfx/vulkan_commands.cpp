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
    VkFenceCreateInfo fi{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    fi.flags = VK_FENCE_CREATE_SIGNALED_BIT; // start signaled
    VK_CHECK(vkCreateFence(dev_, &fi, nullptr, &f.in_flight));

    VkSemaphoreCreateInfo sci{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    VK_CHECK(vkCreateSemaphore(dev_, &sci, nullptr, &f.image_acquired));
    VK_CHECK(vkCreateSemaphore(dev_, &sci, nullptr, &f.render_finished));

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
    if (f.in_flight)      vkDestroyFence(dev_, f.in_flight, nullptr);
    if (f.image_acquired) vkDestroySemaphore(dev_, f.image_acquired, nullptr);
    if (f.render_finished) vkDestroySemaphore(dev_, f.render_finished, nullptr);
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

  // Wait and reset fence for this frame slot
  VK_CHECK(vkWaitForFences(dev_, 1, &f.in_flight, VK_TRUE, UINT64_MAX));
  VK_CHECK(vkResetFences(dev_, 1, &f.in_flight));

  // Acquire image
  uint32_t image_index = 0;
  VkResult acq = vkAcquireNextImageKHR(dev_, swapchain, UINT64_MAX, f.image_acquired, VK_NULL_HANDLE, &image_index);
  if (acq == VK_ERROR_OUT_OF_DATE_KHR) {
    spdlog::warn("[vk] Acquire returned OUT_OF_DATE (resize will recreate).");
    return image_index;
  }
  VK_CHECK(acq);

  // Record commands
  VK_CHECK(vkResetCommandBuffer(f.cmd, 0));
  VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
  VK_CHECK(vkBeginCommandBuffer(f.cmd, &bi));

  // Transition: PRESENT -> COLOR_ATTACHMENT_OPTIMAL (or UNDEFINED on first use)
  VkImageMemoryBarrier to_color{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
  to_color.srcAccessMask = f.first_use ? 0 : VK_ACCESS_MEMORY_READ_BIT;
  to_color.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  to_color.oldLayout = f.first_use ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
  to_color.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  to_color.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_color.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_color.image = swapchain_images[image_index];
  to_color.subresourceRange = full_color_range();

  vkCmdPipelineBarrier(
      f.cmd,
      f.first_use ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      0, 0, nullptr, 0, nullptr, 1, &to_color);

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
  VkImageMemoryBarrier to_present{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
  to_present.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  to_present.dstAccessMask = 0;
  to_present.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  to_present.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
  to_present.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_present.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_present.image = swapchain_images[image_index];
  to_present.subresourceRange = full_color_range();

  vkCmdPipelineBarrier(
      f.cmd,
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
      0, 0, nullptr, 0, nullptr, 1, &to_present);

  VK_CHECK(vkEndCommandBuffer(f.cmd));
  f.first_use = false;

  // Submit
  VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
  si.waitSemaphoreCount = 1;
  si.pWaitSemaphores = &f.image_acquired;
  si.pWaitDstStageMask = &wait_stage;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &f.cmd;
  si.signalSemaphoreCount = 1;
  si.pSignalSemaphores = &f.render_finished;
  
  VK_CHECK(vkQueueSubmit(graphics_queue, 1, &si, f.in_flight));

  // Present
  VkPresentInfoKHR pi{ VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
  pi.waitSemaphoreCount = 1;
  pi.pWaitSemaphores = &f.render_finished;
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

  // Next frame
  frame_cursor_ = (frame_cursor_ + 1) % static_cast<uint32_t>(frames_.size());
  return image_index;
}

} // namespace engine
