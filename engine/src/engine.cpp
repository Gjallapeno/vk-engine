// [UNCHANGED includes...]
#include <engine/engine.hpp>
#include <engine/log.hpp>
#include <engine/config.hpp>
#include <engine/vk_checks.hpp>
#include <engine/platform/window.hpp>
#include <engine/gfx/vulkan_instance.hpp>
#include <engine/gfx/vulkan_surface.hpp>
#include <engine/gfx/vulkan_device.hpp>
#include <engine/gfx/vulkan_swapchain.hpp>
#include <engine/gfx/vulkan_commands.hpp>
#include <engine/gfx/present_pipeline.hpp>
#include <engine/gfx/ray_pipeline.hpp>
#include <engine/gfx/memory.hpp>
#include <engine/camera.hpp>

#include <GLFW/glfw3.h>
#include <algorithm>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <spdlog/spdlog.h>
#include <thread>
#include <chrono>
#include <filesystem>
#include <memory>
#include <vector>
#include <cstdint>
#include <fstream>
#include <glm/gtc/matrix_inverse.hpp>

#ifdef _WIN32
#  define NOMINMAX
#  include <windows.h>
#endif

namespace engine {

static std::filesystem::path exe_dir() {
#ifdef _WIN32
  char buf[MAX_PATH]{};
  DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
  return std::filesystem::path(std::string(buf, buf + n)).parent_path();
#else
  return std::filesystem::current_path();
#endif
}

static constexpr uint32_t kStepsDown = 4;
static constexpr float    kRenderScale = 0.66f;

static float halton(uint32_t i, uint32_t b) {
  float f = 1.0f;
  float r = 0.0f;
  while (i > 0) {
    f /= static_cast<float>(b);
    r += f * static_cast<float>(i % b);
    i /= b;
  }
  return r;
}

struct DrawCtx {
  RayPipeline*      ray_pipe = nullptr;
  VkDescriptorSet   ray_dset = VK_NULL_HANDLE;
  PresentPipeline*  light_pipe = nullptr;
  VkDescriptorSet   light_dset = VK_NULL_HANDLE;
  VkPipeline        comp_pipe = VK_NULL_HANDLE;
  VkPipelineLayout  comp_layout = VK_NULL_HANDLE;
  VkDescriptorSet   comp_set = VK_NULL_HANDLE;
  VkPipeline        comp_l1_pipe = VK_NULL_HANDLE;
  VkPipelineLayout  comp_l1_layout = VK_NULL_HANDLE;
  VkDescriptorSet   comp_l1_set = VK_NULL_HANDLE;
  VkImage           occ_image = VK_NULL_HANDLE;
  VkImage           mat_image = VK_NULL_HANDLE;
  VkImage           occ_l1_image = VK_NULL_HANDLE;
  VkImage           g_albedo = VK_NULL_HANDLE;
  VkImage           g_normal = VK_NULL_HANDLE;
  VkImage           g_depth  = VK_NULL_HANDLE;
  VkImageView       g_albedo_view = VK_NULL_HANDLE;
  VkImageView       g_normal_view = VK_NULL_HANDLE;
  VkImageView       g_depth_view  = VK_NULL_HANDLE;
  VkImage           steps_image = VK_NULL_HANDLE;
  VkImageView       steps_view  = VK_NULL_HANDLE;
  VkBuffer          steps_buffer = VK_NULL_HANDLE;
  VkExtent2D        steps_dim{0,0};
  VkImage           shadow_steps_image = VK_NULL_HANDLE;
  VkImageView       shadow_steps_view  = VK_NULL_HANDLE;
  VkBuffer          shadow_steps_buffer = VK_NULL_HANDLE;
  VkExtent2D        shadow_steps_dim{0,0};
  VkExtent2D        ray_extent{0,0};
  VkExtent3D        occ_dim{0,0,0};
  VkExtent3D        dispatch_dim{0,0,0};
  VkExtent3D        occ_l1_dim{0,0,0};
  VkExtent3D        dispatch_l1_dim{0,0,0};
  bool              first_frame = true;
};

struct CameraUBO {
  glm::mat4 inv_view_proj{1.0f};
  glm::vec2 render_resolution{0.0f};
  glm::vec2 output_resolution{0.0f};
  float     time = 0.0f;
  float     debug_normals = 0.0f;
  float     debug_level = 0.0f;
  float     debug_steps = 0.0f;
  glm::vec4 pad{0.0f};
};

struct VoxelAABB {
  glm::vec3 min{0.0f}; float pad0 = 0.0f;
  glm::vec3 max{0.0f}; float pad1 = 0.0f;
  glm::ivec3 dim{0};   int pad2 = 0;
};

struct VoxParams {
  glm::ivec3 dim{0}; int frame = 0;
  glm::vec3 volMin{0.0f}; float pad0 = 0.0f;
  glm::vec3 volMax{0.0f}; float pad1 = 0.0f;
  glm::vec3 boxCenter{0.0f}; float pad2 = 0.0f;
  glm::vec3 boxHalf{0.0f};   float pad3 = 0.0f;
  glm::vec3 sphereCenter{0.0f}; float sphereRadius = 0.0f;
  int mode = 0; int op = 0; int noiseSeed = 0; int material = 0;
  glm::ivec3 regionMin{0}; int pad5 = 0;
  glm::ivec3 regionMax{0}; float terrainFreq = 0.0f;
};

static VkShaderModule load_module(VkDevice dev, const std::string& path) {
  std::ifstream f(path, std::ios::ate | std::ios::binary);
  if (!f) { spdlog::error("[vk] Failed to open SPIR-V: {}", path); std::abort(); }
  size_t size = static_cast<size_t>(f.tellg());
  std::vector<char> data(size);
  f.seekg(0);
  f.read(data.data(), size);
  VkShaderModuleCreateInfo ci{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
  ci.codeSize = data.size();
  ci.pCode = reinterpret_cast<const uint32_t*>(data.data());
  VkShaderModule mod = VK_NULL_HANDLE;
  VK_CHECK(vkCreateShaderModule(dev, &ci, nullptr, &mod));
  return mod;
}

static void record_present(VkCommandBuffer cmd, VkImage, VkImageView view,
                           VkFormat, VkExtent2D extent, void* user) {
  auto* ctx = static_cast<DrawCtx*>(user);
  VkImageMemoryBarrier pre[3]{ { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER }, { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER }, { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER } };
  for(int i=0;i<3;i++){
    pre[i].srcAccessMask = ctx->first_frame ? 0 : VK_ACCESS_SHADER_READ_BIT;
    pre[i].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    pre[i].oldLayout = ctx->first_frame ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    pre[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    pre[i].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    pre[i].subresourceRange.levelCount = 1;
    pre[i].subresourceRange.layerCount = 1;
  }
  pre[0].image = ctx->occ_image;
  pre[1].image = ctx->mat_image;
  pre[2].image = ctx->occ_l1_image;
  VkPipelineStageFlags srcStage = ctx->first_frame ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                                                   : VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  vkCmdPipelineBarrier(cmd, srcStage, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       0, 0, nullptr, 0, nullptr, 3, pre);

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->comp_pipe);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->comp_layout,
                          0, 1, &ctx->comp_set, 0, nullptr);
  uint32_t gx = (ctx->dispatch_dim.width  + 7) / 8;
  uint32_t gy = (ctx->dispatch_dim.height + 7) / 8;
  uint32_t gz = (ctx->dispatch_dim.depth  + 7) / 8;
  vkCmdDispatch(cmd, gx, gy, gz);

  // Transition L0 outputs for sampling and prepare L1 for writing
  VkImageMemoryBarrier mid[3]{ { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER }, { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER }, { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER } };
  for(int i=0;i<3;i++){
    mid[i].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    mid[i].subresourceRange.levelCount = 1;
    mid[i].subresourceRange.layerCount = 1;
  }
  mid[0].image = ctx->occ_image; mid[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; mid[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  mid[0].oldLayout = VK_IMAGE_LAYOUT_GENERAL; mid[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  mid[1].image = ctx->mat_image; mid[1].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; mid[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  mid[1].oldLayout = VK_IMAGE_LAYOUT_GENERAL; mid[1].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  mid[2].image = ctx->occ_l1_image; mid[2].srcAccessMask = 0; mid[2].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  mid[2].oldLayout = VK_IMAGE_LAYOUT_GENERAL; mid[2].newLayout = VK_IMAGE_LAYOUT_GENERAL;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       0, 0, nullptr, 0, nullptr, 3, mid);

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->comp_l1_pipe);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->comp_l1_layout,
                          0, 1, &ctx->comp_l1_set, 0, nullptr);
  gx = (ctx->dispatch_l1_dim.width  + 7) / 8;
  gy = (ctx->dispatch_l1_dim.height + 7) / 8;
  gz = (ctx->dispatch_l1_dim.depth  + 7) / 8;
  vkCmdDispatch(cmd, gx, gy, gz);

  VkImageMemoryBarrier post[3]{ { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER }, { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER }, { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER } };
  for(int i=0;i<3;i++){
    post[i].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    post[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    post[i].oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    post[i].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    post[i].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    post[i].subresourceRange.levelCount = 1;
    post[i].subresourceRange.layerCount = 1;
  }
  post[0].image = ctx->occ_image;
  post[1].image = ctx->mat_image;
  post[2].image = ctx->occ_l1_image; post[2].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; post[2].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
  post[2].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                       0, 0, nullptr, 0, nullptr, 3, post);

  VkImageMemoryBarrier steps_pre[2]{ { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER }, { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER } };
  for(int i=0;i<2;i++){
    steps_pre[i].srcAccessMask = ctx->first_frame ? 0 : VK_ACCESS_TRANSFER_WRITE_BIT;
    steps_pre[i].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    steps_pre[i].oldLayout = ctx->first_frame ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL;
    steps_pre[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    steps_pre[i].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    steps_pre[i].subresourceRange.levelCount = 1;
    steps_pre[i].subresourceRange.layerCount = 1;
  }
  steps_pre[0].image = ctx->steps_image;
  steps_pre[1].image = ctx->shadow_steps_image;
  VkPipelineStageFlags stepsSrc = ctx->first_frame ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_TRANSFER_BIT;
  vkCmdPipelineBarrier(cmd, stepsSrc, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                       0, 0, nullptr, 0, nullptr, 2, steps_pre);

  // Prepare G-buffer images for color attachment writes
  VkImageMemoryBarrier gpre[3]{ {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}, {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}, {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER} };
  for(int i=0;i<3;i++){
    gpre[i].srcAccessMask = ctx->first_frame ? 0 : VK_ACCESS_SHADER_READ_BIT;
    gpre[i].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    gpre[i].oldLayout = ctx->first_frame ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    gpre[i].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    gpre[i].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    gpre[i].subresourceRange.levelCount = 1;
    gpre[i].subresourceRange.layerCount = 1;
  }
  gpre[0].image = ctx->g_albedo;
  gpre[1].image = ctx->g_normal;
  gpre[2].image = ctx->g_depth;
  VkPipelineStageFlags gpreSrc = ctx->first_frame ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  vkCmdPipelineBarrier(cmd, gpreSrc, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                       0, 0, nullptr, 0, nullptr, 3, gpre);

  // Geometry pass writing G-buffer
  VkRenderingAttachmentInfo gAtt[3]{ {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO}, {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO}, {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO} };
  VkClearValue gclear{}; gclear.color = {{0,0,0,0}};
  gAtt[0].imageView = ctx->g_albedo_view; gAtt[0].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  gAtt[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; gAtt[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE; gAtt[0].clearValue = gclear;
  gAtt[1].imageView = ctx->g_normal_view; gAtt[1].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  gAtt[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; gAtt[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE; gAtt[1].clearValue = gclear;
  gAtt[2].imageView = ctx->g_depth_view; gAtt[2].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  gAtt[2].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; gAtt[2].storeOp = VK_ATTACHMENT_STORE_OP_STORE; gAtt[2].clearValue = gclear;

  VkRenderingInfo gi{ VK_STRUCTURE_TYPE_RENDERING_INFO };
  gi.renderArea.offset = {0,0};
  gi.renderArea.extent = ctx->ray_extent;
  gi.layerCount = 1;
  gi.colorAttachmentCount = 3;
  gi.pColorAttachments = gAtt;

  vkCmdBeginRendering(cmd, &gi);

  // Geometry pass renders the voxel scene at reduced resolution (ray_extent)
  VkViewport vp_lo{};
  vp_lo.x = 0.0f;
  vp_lo.y = static_cast<float>(ctx->ray_extent.height);
  vp_lo.width  = static_cast<float>(ctx->ray_extent.width);
  vp_lo.height = -static_cast<float>(ctx->ray_extent.height);
  vp_lo.minDepth = 0.0f;
  vp_lo.maxDepth = 1.0f;
  vkCmdSetViewport(cmd, 0, 1, &vp_lo);
  VkRect2D sc_lo{ {0,0}, ctx->ray_extent };
  vkCmdSetScissor(cmd, 0, 1, &sc_lo);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, ctx->ray_pipe->pipeline());
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, ctx->ray_pipe->layout(),
                          0, 1, &ctx->ray_dset, 0, nullptr);
  vkCmdDraw(cmd, 3, 1, 0, 0);
  vkCmdEndRendering(cmd);

  VkImageSubresourceRange stepsRange{};
  stepsRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  stepsRange.levelCount = 1;
  stepsRange.layerCount = 1;

  VkImageMemoryBarrier steps_to_copy{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
  steps_to_copy.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  steps_to_copy.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  steps_to_copy.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
  steps_to_copy.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  steps_to_copy.subresourceRange = stepsRange;
  steps_to_copy.image = ctx->steps_image;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT,
                       0, 0, nullptr, 0, nullptr, 1, &steps_to_copy);

  VkBufferImageCopy bic{};
  bic.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  bic.imageSubresource.layerCount = 1;
  bic.imageExtent = { ctx->steps_dim.width, ctx->steps_dim.height, 1 };
  vkCmdCopyImageToBuffer(cmd, ctx->steps_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         ctx->steps_buffer, 1, &bic);

  VkImageMemoryBarrier steps_to_clear{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
  steps_to_clear.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  steps_to_clear.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  steps_to_clear.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  steps_to_clear.newLayout = VK_IMAGE_LAYOUT_GENERAL;
  steps_to_clear.subresourceRange = stepsRange;
  steps_to_clear.image = ctx->steps_image;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       0, 0, nullptr, 0, nullptr, 1, &steps_to_clear);

  VkClearColorValue zero{{0,0,0,0}};
  vkCmdClearColorImage(cmd, ctx->steps_image, VK_IMAGE_LAYOUT_GENERAL,
                       &zero, 1, &stepsRange);

  // Transition G-buffer for sampling
  VkImageMemoryBarrier gpost[3]{ {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}, {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}, {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER} };
  for(int i=0;i<3;i++){
    gpost[i].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    gpost[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    gpost[i].oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    gpost[i].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    gpost[i].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    gpost[i].subresourceRange.levelCount = 1;
    gpost[i].subresourceRange.layerCount = 1;
  }
  gpost[0].image = ctx->g_albedo;
  gpost[1].image = ctx->g_normal;
  gpost[2].image = ctx->g_depth;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                       0, 0, nullptr, 0, nullptr, 3, gpost);

  // Lighting pass to final image
  VkRenderingAttachmentInfo color{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
  color.imageView   = view;
  color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  color.loadOp  = VK_ATTACHMENT_LOAD_OP_CLEAR;
  color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  VkClearValue clear; clear.color = { {0.06f, 0.07f, 0.10f, 1.0f} };
  color.clearValue = clear;

  VkRenderingInfo ri{ VK_STRUCTURE_TYPE_RENDERING_INFO };
  ri.renderArea.offset = {0, 0};
  ri.renderArea.extent = extent;
  ri.layerCount = 1;
  ri.colorAttachmentCount = 1;
  ri.pColorAttachments = &color;

  // Final lighting/present pass upscales to the swapchain extent
  vkCmdBeginRendering(cmd, &ri);

  VkViewport vp_hi{};
  vp_hi.x = 0.0f;
  vp_hi.y = static_cast<float>(extent.height);
  vp_hi.width  = static_cast<float>(extent.width);
  vp_hi.height = -static_cast<float>(extent.height);
  vp_hi.minDepth = 0.0f;
  vp_hi.maxDepth = 1.0f;
  vkCmdSetViewport(cmd, 0, 1, &vp_hi);

  VkRect2D sc_hi{ {0,0}, extent };
  vkCmdSetScissor(cmd, 0, 1, &sc_hi);

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, ctx->light_pipe->pipeline());
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, ctx->light_pipe->layout(),
                          0, 1, &ctx->light_dset, 0, nullptr);
  vkCmdDraw(cmd, 3, 1, 0, 0);
  vkCmdEndRendering(cmd);

  VkImageMemoryBarrier sh_to_copy{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
  sh_to_copy.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  sh_to_copy.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  sh_to_copy.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
  sh_to_copy.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  sh_to_copy.subresourceRange = stepsRange;
  sh_to_copy.image = ctx->shadow_steps_image;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       0, 0, nullptr, 0, nullptr, 1, &sh_to_copy);

  VkBufferImageCopy bic2{};
  bic2.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  bic2.imageSubresource.layerCount = 1;
  bic2.imageExtent = { ctx->shadow_steps_dim.width, ctx->shadow_steps_dim.height, 1 };
  vkCmdCopyImageToBuffer(cmd, ctx->shadow_steps_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         ctx->shadow_steps_buffer, 1, &bic2);

  VkImageMemoryBarrier sh_to_clear{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
  sh_to_clear.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  sh_to_clear.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  sh_to_clear.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  sh_to_clear.newLayout = VK_IMAGE_LAYOUT_GENERAL;
  sh_to_clear.subresourceRange = stepsRange;
  sh_to_clear.image = ctx->shadow_steps_image;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       0, 0, nullptr, 0, nullptr, 1, &sh_to_clear);

  vkCmdClearColorImage(cmd, ctx->shadow_steps_image, VK_IMAGE_LAYOUT_GENERAL,
                       &zero, 1, &stepsRange);

  ctx->first_frame = false;
}

int run() {
  init_logging();
  log_boot_banner("engine");

  WindowDesc wdesc{}; wdesc.title = "vk_engine fullscreen present";
  auto window = create_window(wdesc);

  Camera cam;
  GLFWwindow* glfw_win = static_cast<GLFWwindow*>(window->native_handle());
  glfwSetInputMode(glfw_win, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
  double last_x = 0.0, last_y = 0.0;
  glfwGetCursorPos(glfw_win, &last_x, &last_y);
  auto last_time = std::chrono::high_resolution_clock::now();

  VulkanInstanceCreateInfo ici{}; ici.enable_validation = cfg::kValidation;
  for (auto* e : platform_required_instance_extensions()) ici.extra_extensions.push_back(e);
  VulkanInstance instance{ici};
  VulkanSurface surface{instance.vk(), window->native_handle()};
  VulkanDeviceCreateInfo dci{}; dci.instance = instance.vk(); dci.surface = surface.vk(); dci.enable_validation = cfg::kValidation;
  VulkanDevice device{dci};

  const auto shader_dir = exe_dir() / "shaders";
  const auto vs_path = (shader_dir / "vs_fullscreen.vert.spv").string();
  const auto ray_fs_path = (shader_dir / "fs_raycast.frag.spv").string();
  const auto light_fs_path = (shader_dir / "fs_lighting.frag.spv").string();
  spdlog::info("[vk] Using shaders: {}", shader_dir.string());

  GpuAllocator allocator; allocator.init(instance.vk(), device.physical(), device.device());

  // Samplers for sampled images
  VkSampler linear_sampler  = VK_NULL_HANDLE;
  VkSampler nearest_sampler = VK_NULL_HANDLE;
  {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(device.physical(), &props);
    VkPhysicalDeviceFeatures feats{};
    vkGetPhysicalDeviceFeatures(device.physical(), &feats);

    VkSamplerCreateInfo si{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.minLod = 0.0f; si.maxLod = 0.0f;

    // linear sampler for floating point images
    si.magFilter = VK_FILTER_LINEAR; si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    if (feats.samplerAnisotropy) {
      si.anisotropyEnable = VK_TRUE;
      si.maxAnisotropy = props.limits.maxSamplerAnisotropy;
    } else {
      si.anisotropyEnable = VK_FALSE;
      si.maxAnisotropy = 1.0f;
    }
    VK_CHECK(vkCreateSampler(device.device(), &si, nullptr, &linear_sampler));

    // nearest sampler for integer textures (e.g. occupancy grid)
    si.magFilter = VK_FILTER_NEAREST; si.minFilter = VK_FILTER_NEAREST;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.anisotropyEnable = VK_FALSE; si.maxAnisotropy = 1.0f;
    VK_CHECK(vkCreateSampler(device.device(), &si, nullptr, &nearest_sampler));
  }

  // Buffers and occupancy texture
  Buffer cam_buf = create_buffer(allocator.raw(), sizeof(CameraUBO),
                                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
  Buffer vox_buf = create_buffer(allocator.raw(), sizeof(VoxelAABB),
                                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
  Buffer vox_params_buf = create_buffer(allocator.raw(), sizeof(VoxParams),
                                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
  const uint32_t N = 128;
  Image3D occ_img{};
  Image3D mat_img{};
  Image3D occ_l1_img{};
  VkImageView occ_view = VK_NULL_HANDLE;
  VkImageView occ_storage_view = VK_NULL_HANDLE;
  VkImageView occ_l1_view = VK_NULL_HANDLE;
  VkImageView occ_l1_storage_view = VK_NULL_HANDLE;
  VkImageView mat_view = VK_NULL_HANDLE;
  VkImageView mat_storage_view = VK_NULL_HANDLE;
  {
    occ_img = create_image3d(allocator.raw(), N, N, N,
                             VK_FORMAT_R8_UINT,
                             VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    VkImageViewCreateInfo ovi{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    ovi.image = occ_img.image; ovi.viewType = VK_IMAGE_VIEW_TYPE_3D;
    ovi.format = VK_FORMAT_R8_UINT;
    ovi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    ovi.subresourceRange.levelCount = 1;
    ovi.subresourceRange.layerCount = 1;
    VK_CHECK(vkCreateImageView(device.device(), &ovi, nullptr, &occ_view));
    VK_CHECK(vkCreateImageView(device.device(), &ovi, nullptr, &occ_storage_view));

    const uint32_t N1 = N/4;
    occ_l1_img = create_image3d(allocator.raw(), N1, N1, N1,
                                VK_FORMAT_R8_UINT,
                                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    ovi.image = occ_l1_img.image;
    VK_CHECK(vkCreateImageView(device.device(), &ovi, nullptr, &occ_l1_view));
    VK_CHECK(vkCreateImageView(device.device(), &ovi, nullptr, &occ_l1_storage_view));

    mat_img = create_image3d(allocator.raw(), N, N, N,
                             VK_FORMAT_R8_UINT,
                             VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    ovi.image = mat_img.image;
    VK_CHECK(vkCreateImageView(device.device(), &ovi, nullptr, &mat_view));
    VK_CHECK(vkCreateImageView(device.device(), &ovi, nullptr, &mat_storage_view));
  }

  // Upload static voxel bounds
  VoxelAABB vubo{};
  vubo.min = {0.0f,0.0f,0.0f};
  vubo.max = {static_cast<float>(N),static_cast<float>(N),static_cast<float>(N)};
  vubo.dim = {static_cast<int>(N),static_cast<int>(N),static_cast<int>(N)};
  upload_buffer(allocator.raw(), device.device(), device.graphics_family(),
                device.graphics_queue(), vox_buf, &vubo, sizeof(vubo));

  // Compute pipeline to generate voxel occupancy and material textures
  VkDescriptorSetLayout comp_dsl = VK_NULL_HANDLE;
  VkPipelineLayout      comp_layout = VK_NULL_HANDLE;
  VkPipeline            comp_pipeline = VK_NULL_HANDLE;
  VkDescriptorPool      comp_pool = VK_NULL_HANDLE;
  VkDescriptorSet       comp_set  = VK_NULL_HANDLE;
  VkDescriptorSetLayout comp_l1_dsl = VK_NULL_HANDLE;
  VkPipelineLayout      comp_l1_layout = VK_NULL_HANDLE;
  VkPipeline            comp_l1_pipeline = VK_NULL_HANDLE;
  VkDescriptorPool      comp_l1_pool = VK_NULL_HANDLE;
  VkDescriptorSet       comp_l1_set  = VK_NULL_HANDLE;
  {
    VkDescriptorSetLayoutBinding binds[3]{};
    binds[0].binding = 0; binds[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    binds[0].descriptorCount = 1; binds[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    binds[1].binding = 1; binds[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    binds[1].descriptorCount = 1; binds[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    binds[2].binding = 2; binds[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    binds[2].descriptorCount = 1; binds[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dlci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    dlci.bindingCount = 3; dlci.pBindings = binds;
    VK_CHECK(vkCreateDescriptorSetLayout(device.device(), &dlci, nullptr, &comp_dsl));

    VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1; plci.pSetLayouts = &comp_dsl;
    VK_CHECK(vkCreatePipelineLayout(device.device(), &plci, nullptr, &comp_layout));

    const auto cs_path = (shader_dir / "procgen_voxels.comp.spv").string();
    VkShaderModule cs = load_module(device.device(), cs_path);
    VkPipelineShaderStageCreateInfo stage{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = cs;
    stage.pName  = "main";
    VkComputePipelineCreateInfo cpci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpci.stage = stage; cpci.layout = comp_layout;
    VK_CHECK(vkCreateComputePipelines(device.device(), device.pipeline_cache(), 1, &cpci, nullptr, &comp_pipeline));
    vkDestroyShaderModule(device.device(), cs, nullptr);

    VkDescriptorPoolSize psizes[2]{};
    psizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; psizes[0].descriptorCount = 2;
    psizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; psizes[1].descriptorCount = 1;
    VkDescriptorPoolCreateInfo dpci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpci.maxSets = 1; dpci.poolSizeCount = 2; dpci.pPoolSizes = psizes;
    VK_CHECK(vkCreateDescriptorPool(device.device(), &dpci, nullptr, &comp_pool));

    VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    ai.descriptorPool = comp_pool; ai.descriptorSetCount = 1; ai.pSetLayouts = &comp_dsl;
    VK_CHECK(vkAllocateDescriptorSets(device.device(), &ai, &comp_set));

    VkDescriptorImageInfo occ_info{}; occ_info.imageView = occ_storage_view; occ_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorImageInfo mat_info{}; mat_info.imageView = mat_storage_view; mat_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorBufferInfo param_bi{};
    param_bi.buffer = vox_params_buf.buffer; param_bi.range = sizeof(VoxParams);
    VkWriteDescriptorSet ws[3]{};
    ws[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    ws[0].dstSet = comp_set; ws[0].dstBinding = 0;
    ws[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    ws[0].descriptorCount = 1; ws[0].pImageInfo = &occ_info;
    ws[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    ws[1].dstSet = comp_set; ws[1].dstBinding = 1;
    ws[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    ws[1].descriptorCount = 1; ws[1].pImageInfo = &mat_info;
    ws[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    ws[2].dstSet = comp_set; ws[2].dstBinding = 2;
    ws[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    ws[2].descriptorCount = 1; ws[2].pBufferInfo = &param_bi;
    vkUpdateDescriptorSets(device.device(), 3, ws, 0, nullptr);
  }

  // Compute pipeline to build coarse L1 occupancy from L0
  {
    VkDescriptorSetLayoutBinding binds[2]{};
    binds[0].binding = 0; binds[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    binds[0].descriptorCount = 1; binds[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    binds[1].binding = 1; binds[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    binds[1].descriptorCount = 1; binds[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dlci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    dlci.bindingCount = 2; dlci.pBindings = binds;
    VK_CHECK(vkCreateDescriptorSetLayout(device.device(), &dlci, nullptr, &comp_l1_dsl));

    VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1; plci.pSetLayouts = &comp_l1_dsl;
    VK_CHECK(vkCreatePipelineLayout(device.device(), &plci, nullptr, &comp_l1_layout));

    const auto cs_path = (shader_dir / "build_occ_l1.comp.spv").string();
    VkShaderModule cs = load_module(device.device(), cs_path);
    VkPipelineShaderStageCreateInfo stage{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = cs;
    stage.pName = "main";
    VkComputePipelineCreateInfo cpci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpci.stage = stage; cpci.layout = comp_l1_layout;
    VK_CHECK(vkCreateComputePipelines(device.device(), device.pipeline_cache(), 1, &cpci, nullptr, &comp_l1_pipeline));
    vkDestroyShaderModule(device.device(), cs, nullptr);

    VkDescriptorPoolSize psizes[2]{};
    psizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; psizes[0].descriptorCount = 1;
    psizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; psizes[1].descriptorCount = 1;
    VkDescriptorPoolCreateInfo dpci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpci.maxSets = 1; dpci.poolSizeCount = 2; dpci.pPoolSizes = psizes;
    VK_CHECK(vkCreateDescriptorPool(device.device(), &dpci, nullptr, &comp_l1_pool));

    VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    ai.descriptorPool = comp_l1_pool; ai.descriptorSetCount = 1; ai.pSetLayouts = &comp_l1_dsl;
    VK_CHECK(vkAllocateDescriptorSets(device.device(), &ai, &comp_l1_set));

    VkDescriptorImageInfo occ0_info{}; occ0_info.sampler = nearest_sampler; occ0_info.imageView = occ_view;
    occ0_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo occ1_info{}; occ1_info.imageView = occ_l1_storage_view;
    occ1_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkWriteDescriptorSet ws1[2]{};
    ws1[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    ws1[0].dstSet = comp_l1_set; ws1[0].dstBinding = 0;
    ws1[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    ws1[0].descriptorCount = 1; ws1[0].pImageInfo = &occ0_info;
    ws1[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    ws1[1].dstSet = comp_l1_set; ws1[1].dstBinding = 1;
    ws1[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    ws1[1].descriptorCount = 1; ws1[1].pImageInfo = &occ1_info;
    vkUpdateDescriptorSets(device.device(), 2, ws1, 0, nullptr);
  }

  std::unique_ptr<VulkanSwapchain>   swapchain;
  std::unique_ptr<VulkanCommands>    commands;
  std::unique_ptr<RayPipeline>       ray_pipeline;
  std::unique_ptr<PresentPipeline>   present_pipeline;

  VkDescriptorPool dpool = VK_NULL_HANDLE;
  VkDescriptorSet  ray_dset  = VK_NULL_HANDLE;
  VkDescriptorSet  light_dset = VK_NULL_HANDLE;

  DrawCtx ctx{};

  // Descriptor pool for geometry and lighting passes
  {
    VkDescriptorPoolSize sizes[3]{};
    sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; sizes[0].descriptorCount = 4;
    sizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; sizes[1].descriptorCount = 8;
    sizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; sizes[2].descriptorCount = 2;
    VkDescriptorPoolCreateInfo dpci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpci.maxSets = 2; dpci.poolSizeCount = 3; dpci.pPoolSizes = sizes;
    VK_CHECK(vkCreateDescriptorPool(device.device(), &dpci, nullptr, &dpool));
  }

  Image2D g_albedo_img{};
  Image2D g_normal_img{};
  Image2D g_depth_img{};
  Image2D steps_img{};
  Image2D shadow_steps_img{};
  VkImageView g_albedo_view = VK_NULL_HANDLE;
  VkImageView g_normal_view = VK_NULL_HANDLE;
  VkImageView g_depth_view = VK_NULL_HANDLE;
  VkImageView steps_view = VK_NULL_HANDLE;
  VkImageView shadow_steps_view = VK_NULL_HANDLE;
  Buffer steps_buf{};
  Buffer shadow_steps_buf{};

  auto destroy_swapchain_stack = [&]() {
    if (ray_dset) { vkFreeDescriptorSets(device.device(), dpool, 1, &ray_dset); ray_dset = VK_NULL_HANDLE; }
    if (light_dset) { vkFreeDescriptorSets(device.device(), dpool, 1, &light_dset); light_dset = VK_NULL_HANDLE; }
    if (g_albedo_view) vkDestroyImageView(device.device(), g_albedo_view, nullptr);
    if (g_normal_view) vkDestroyImageView(device.device(), g_normal_view, nullptr);
    if (g_depth_view) vkDestroyImageView(device.device(), g_depth_view, nullptr);
    if (steps_view) vkDestroyImageView(device.device(), steps_view, nullptr);
    if (shadow_steps_view) vkDestroyImageView(device.device(), shadow_steps_view, nullptr);
    destroy_image2d(allocator.raw(), g_albedo_img); destroy_image2d(allocator.raw(), g_normal_img); destroy_image2d(allocator.raw(), g_depth_img);
    destroy_image2d(allocator.raw(), steps_img); destroy_buffer(allocator.raw(), steps_buf);
    destroy_image2d(allocator.raw(), shadow_steps_img); destroy_buffer(allocator.raw(), shadow_steps_buf);
    commands.reset();
    swapchain.reset();
  };

  auto create_swapchain_stack = [&](uint32_t sw, uint32_t sh)
  {
    VulkanSwapchainCreateInfo sci{};
    sci.physical = device.physical(); sci.device = device.device(); sci.surface = surface.vk();
    sci.desired_width = sw; sci.desired_height = sh;
    swapchain = std::make_unique<VulkanSwapchain>(sci);

    VulkanCommandsCreateInfo cci{};
    cci.device = device.device(); cci.graphics_family = device.graphics_family();
    cci.image_count = static_cast<uint32_t>(swapchain->image_views().size());
    commands = std::make_unique<VulkanCommands>(cci);

    if (!ray_pipeline) {
      RayPipelineCreateInfo rpci{}; rpci.device = device.device(); rpci.pipeline_cache = device.pipeline_cache();
      rpci.vs_spv = vs_path; rpci.fs_spv = ray_fs_path;
      ray_pipeline = std::make_unique<RayPipeline>(rpci);
    }
    if (!present_pipeline || present_pipeline->color_format() != swapchain->image_format()) {
      PresentPipelineCreateInfo pci{}; pci.device = device.device();
      pci.pipeline_cache = device.pipeline_cache(); pci.color_format = swapchain->image_format();
      pci.vs_spv = vs_path; pci.fs_spv = light_fs_path;
      present_pipeline = std::make_unique<PresentPipeline>(pci);
    }

    if (ray_dset) { vkFreeDescriptorSets(device.device(), dpool, 1, &ray_dset); ray_dset = VK_NULL_HANDLE; }
    if (light_dset) { vkFreeDescriptorSets(device.device(), dpool, 1, &light_dset); light_dset = VK_NULL_HANDLE; }

    VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    ai.descriptorPool = dpool;
    VkDescriptorSetLayout layout = ray_pipeline->dset_layout();
    ai.descriptorSetCount = 1; ai.pSetLayouts = &layout;
    VK_CHECK(vkAllocateDescriptorSets(device.device(), &ai, &ray_dset));

    uint32_t rw = static_cast<uint32_t>(static_cast<float>(sw) * kRenderScale);
    uint32_t rh = static_cast<uint32_t>(static_cast<float>(sh) * kRenderScale);
    rw = std::max(1u, rw); rh = std::max(1u, rh);
    uint32_t steps_w = (rw + kStepsDown - 1) / kStepsDown;
    uint32_t steps_h = (rh + kStepsDown - 1) / kStepsDown;
    steps_img = create_image2d(allocator.raw(), steps_w, steps_h, VK_FORMAT_R32_UINT,
                               VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    VkImageViewCreateInfo svi{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    svi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    svi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    svi.subresourceRange.levelCount = 1;
    svi.subresourceRange.layerCount = 1;
    svi.image = steps_img.image; svi.format = VK_FORMAT_R32_UINT;
    VK_CHECK(vkCreateImageView(device.device(), &svi, nullptr, &steps_view));
    steps_buf = create_host_buffer(allocator.raw(), steps_w * steps_h * sizeof(uint32_t),
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    shadow_steps_img = create_image2d(allocator.raw(), steps_w, steps_h, VK_FORMAT_R32_UINT,
                                      VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    svi.image = shadow_steps_img.image; svi.format = VK_FORMAT_R32_UINT;
    VK_CHECK(vkCreateImageView(device.device(), &svi, nullptr, &shadow_steps_view));
    shadow_steps_buf = create_host_buffer(allocator.raw(), steps_w * steps_h * sizeof(uint32_t),
                                         VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    VkDescriptorBufferInfo cam_bi{}; cam_bi.buffer = cam_buf.buffer; cam_bi.range = sizeof(CameraUBO);
    VkDescriptorBufferInfo vox_bi{}; vox_bi.buffer = vox_buf.buffer; vox_bi.range = sizeof(VoxelAABB);
    VkDescriptorImageInfo occ_info{}; occ_info.sampler = nearest_sampler; occ_info.imageView = occ_view; occ_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo mat_info{}; mat_info.sampler = nearest_sampler; mat_info.imageView = mat_view; mat_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo occ_l1_info{}; occ_l1_info.sampler = nearest_sampler; occ_l1_info.imageView = occ_l1_view; occ_l1_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo steps_info{}; steps_info.imageView = steps_view; steps_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet rwrites[6]{};
    rwrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; rwrites[0].dstSet = ray_dset; rwrites[0].dstBinding = 0; rwrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; rwrites[0].descriptorCount = 1; rwrites[0].pBufferInfo = &cam_bi;
    rwrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; rwrites[1].dstSet = ray_dset; rwrites[1].dstBinding = 1; rwrites[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; rwrites[1].descriptorCount = 1; rwrites[1].pBufferInfo = &vox_bi;
    rwrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; rwrites[2].dstSet = ray_dset; rwrites[2].dstBinding = 2; rwrites[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; rwrites[2].descriptorCount = 1; rwrites[2].pImageInfo = &occ_info;
    rwrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; rwrites[3].dstSet = ray_dset; rwrites[3].dstBinding = 3; rwrites[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; rwrites[3].descriptorCount = 1; rwrites[3].pImageInfo = &mat_info;
    rwrites[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; rwrites[4].dstSet = ray_dset; rwrites[4].dstBinding = 4; rwrites[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; rwrites[4].descriptorCount = 1; rwrites[4].pImageInfo = &occ_l1_info;
    rwrites[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; rwrites[5].dstSet = ray_dset; rwrites[5].dstBinding = 5; rwrites[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; rwrites[5].descriptorCount = 1; rwrites[5].pImageInfo = &steps_info;
    vkUpdateDescriptorSets(device.device(), 6, rwrites, 0, nullptr);

    g_albedo_img = create_image2d(allocator.raw(), rw, rh, VK_FORMAT_R8G8B8A8_UNORM,
                                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    g_normal_img = create_image2d(allocator.raw(), rw, rh, VK_FORMAT_R16G16B16A16_SFLOAT,
                                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    g_depth_img = create_image2d(allocator.raw(), rw, rh, VK_FORMAT_R32_SFLOAT,
                                 VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

    VkImageViewCreateInfo iv{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    iv.viewType = VK_IMAGE_VIEW_TYPE_2D; iv.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; iv.subresourceRange.levelCount = 1; iv.subresourceRange.layerCount = 1;
    iv.image = g_albedo_img.image; iv.format = VK_FORMAT_R8G8B8A8_UNORM; VK_CHECK(vkCreateImageView(device.device(), &iv, nullptr, &g_albedo_view));
    iv.image = g_normal_img.image; iv.format = VK_FORMAT_R16G16B16A16_SFLOAT; VK_CHECK(vkCreateImageView(device.device(), &iv, nullptr, &g_normal_view));
    iv.image = g_depth_img.image; iv.format = VK_FORMAT_R32_SFLOAT; VK_CHECK(vkCreateImageView(device.device(), &iv, nullptr, &g_depth_view));

    VkDescriptorSetLayout layout2 = present_pipeline->dset_layout();
    ai.pSetLayouts = &layout2;
    VK_CHECK(vkAllocateDescriptorSets(device.device(), &ai, &light_dset));

    VkDescriptorImageInfo albedo_info{}; albedo_info.sampler = linear_sampler; albedo_info.imageView = g_albedo_view; albedo_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo normal_info{}; normal_info.sampler = linear_sampler; normal_info.imageView = g_normal_view; normal_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo depth_info{}; depth_info.sampler = linear_sampler; depth_info.imageView = g_depth_view; depth_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo shadow_steps_info{}; shadow_steps_info.imageView = shadow_steps_view; shadow_steps_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet lwrites[8]{};
    lwrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; lwrites[0].dstSet = light_dset; lwrites[0].dstBinding = 0; lwrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; lwrites[0].descriptorCount = 1; lwrites[0].pBufferInfo = &cam_bi;
    lwrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; lwrites[1].dstSet = light_dset; lwrites[1].dstBinding = 1; lwrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; lwrites[1].descriptorCount = 1; lwrites[1].pImageInfo = &albedo_info;
    lwrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; lwrites[2].dstSet = light_dset; lwrites[2].dstBinding = 2; lwrites[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; lwrites[2].descriptorCount = 1; lwrites[2].pImageInfo = &normal_info;
    lwrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; lwrites[3].dstSet = light_dset; lwrites[3].dstBinding = 3; lwrites[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; lwrites[3].descriptorCount = 1; lwrites[3].pImageInfo = &depth_info;
    lwrites[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; lwrites[4].dstSet = light_dset; lwrites[4].dstBinding = 4; lwrites[4].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; lwrites[4].descriptorCount = 1; lwrites[4].pBufferInfo = &vox_bi;
    lwrites[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; lwrites[5].dstSet = light_dset; lwrites[5].dstBinding = 5; lwrites[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; lwrites[5].descriptorCount = 1; lwrites[5].pImageInfo = &occ_info;
    lwrites[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; lwrites[6].dstSet = light_dset; lwrites[6].dstBinding = 6; lwrites[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; lwrites[6].descriptorCount = 1; lwrites[6].pImageInfo = &occ_l1_info;
    lwrites[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; lwrites[7].dstSet = light_dset; lwrites[7].dstBinding = 7; lwrites[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; lwrites[7].descriptorCount = 1; lwrites[7].pImageInfo = &shadow_steps_info;
    vkUpdateDescriptorSets(device.device(), 8, lwrites, 0, nullptr);

    ctx.ray_pipe = ray_pipeline.get(); ctx.ray_dset = ray_dset;
    ctx.light_pipe = present_pipeline.get(); ctx.light_dset = light_dset;
    ctx.g_albedo = g_albedo_img.image; ctx.g_normal = g_normal_img.image; ctx.g_depth = g_depth_img.image;
    ctx.g_albedo_view = g_albedo_view; ctx.g_normal_view = g_normal_view; ctx.g_depth_view = g_depth_view;
    ctx.steps_image = steps_img.image; ctx.steps_view = steps_view; ctx.steps_buffer = steps_buf.buffer; ctx.steps_dim = {steps_w, steps_h};
    ctx.shadow_steps_image = shadow_steps_img.image; ctx.shadow_steps_view = shadow_steps_view; ctx.shadow_steps_buffer = shadow_steps_buf.buffer; ctx.shadow_steps_dim = {steps_w, steps_h};
    ctx.ray_extent = {rw, rh};
    ctx.first_frame = true;
  };

  { auto fb = window->framebuffer_size();
    create_swapchain_stack(static_cast<uint32_t>(fb.first), static_cast<uint32_t>(fb.second)); }

  ctx.comp_pipe = comp_pipeline; ctx.comp_layout = comp_layout; ctx.comp_set = comp_set;
  ctx.comp_l1_pipe = comp_l1_pipeline; ctx.comp_l1_layout = comp_l1_layout; ctx.comp_l1_set = comp_l1_set;
  ctx.occ_image = occ_img.image; ctx.mat_image = mat_img.image; ctx.occ_l1_image = occ_l1_img.image;
  ctx.occ_dim = {N, N, N}; ctx.dispatch_dim = {N, N, N};
  ctx.occ_l1_dim = {N/4, N/4, N/4}; ctx.dispatch_l1_dim = {N/4, N/4, N/4};
  ctx.first_frame = true;
  VkExtent2D last = swapchain->extent();
  float total_time = 0.0f;
  int   frame_counter = 0;
  uint32_t jitter_index = 0;
  using namespace std::chrono_literals;

  enum { MODE_CLEAR=0, MODE_FILL_BOX=1, MODE_FILL_SPHERE=2, MODE_NOISE=3, MODE_TERRAIN=4 };
  enum { OP_REPLACE=0, OP_UNION=1, OP_INTERSECTION=2, OP_SUBTRACT=3 };

  glm::vec3 sphere_center{static_cast<float>(N)/2.0f,
                          static_cast<float>(N)/2.0f,
                          static_cast<float>(N)/2.0f};
  float sphere_radius = 30.0f;
  glm::vec3 box_center{60.0f,60.0f,60.0f};
  glm::vec3 box_half{40.0f,40.0f,40.0f};
  int vox_mode = MODE_TERRAIN;
  int vox_op = OP_REPLACE;
  int noise_seed = 0;

  while (!window->should_close()) {
    window->poll_events();

    auto now = std::chrono::high_resolution_clock::now();
    float dt = std::chrono::duration<float>(now - last_time).count();
    last_time = now;
    total_time += dt;

    double cx, cy;
    glfwGetCursorPos(glfw_win, &cx, &cy);
    float dx = static_cast<float>(cx - last_x);
    float dy = static_cast<float>(cy - last_y);
    last_x = cx; last_y = cy;

    const float sens = 0.002f;
    cam.yaw   += dx * sens;
    cam.pitch -= dy * sens;
    cam.pitch = std::clamp(cam.pitch, -glm::half_pi<float>() + 0.01f,
                                      glm::half_pi<float>() - 0.01f);

    glm::vec3 move{0.0f};
    if (glfwGetKey(glfw_win, GLFW_KEY_W) == GLFW_PRESS) move += cam.forward();
    if (glfwGetKey(glfw_win, GLFW_KEY_S) == GLFW_PRESS) move -= cam.forward();
    if (glfwGetKey(glfw_win, GLFW_KEY_A) == GLFW_PRESS) move -= cam.right();
    if (glfwGetKey(glfw_win, GLFW_KEY_D) == GLFW_PRESS) move += cam.right();
    if (glfwGetKey(glfw_win, GLFW_KEY_SPACE) == GLFW_PRESS) move += glm::vec3(0.0f,1.0f,0.0f);
    if (glfwGetKey(glfw_win, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) move -= glm::vec3(0.0f,1.0f,0.0f);
    if (glm::length(move) > 0.0f) {
      cam.position += glm::normalize(move) * (2.0f * dt);
    }

    // Voxel editing hotkeys
    if (glfwGetKey(glfw_win, GLFW_KEY_1) == GLFW_PRESS) vox_mode = MODE_CLEAR;
    if (glfwGetKey(glfw_win, GLFW_KEY_2) == GLFW_PRESS) vox_mode = MODE_FILL_BOX;
    if (glfwGetKey(glfw_win, GLFW_KEY_3) == GLFW_PRESS) vox_mode = MODE_FILL_SPHERE;
    if (glfwGetKey(glfw_win, GLFW_KEY_4) == GLFW_PRESS) vox_mode = MODE_NOISE;

    if (glfwGetKey(glfw_win, GLFW_KEY_5) == GLFW_PRESS) vox_op = OP_REPLACE;
    if (glfwGetKey(glfw_win, GLFW_KEY_6) == GLFW_PRESS) vox_op = OP_UNION;
    if (glfwGetKey(glfw_win, GLFW_KEY_7) == GLFW_PRESS) vox_op = OP_INTERSECTION;
    if (glfwGetKey(glfw_win, GLFW_KEY_8) == GLFW_PRESS) vox_op = OP_SUBTRACT;

    float spd = 30.0f * dt;
    if (vox_mode == MODE_FILL_SPHERE) {
      if (glfwGetKey(glfw_win, GLFW_KEY_LEFT) == GLFW_PRESS)  sphere_center.x -= spd;
      if (glfwGetKey(glfw_win, GLFW_KEY_RIGHT) == GLFW_PRESS) sphere_center.x += spd;
      if (glfwGetKey(glfw_win, GLFW_KEY_UP) == GLFW_PRESS)    sphere_center.y += spd;
      if (glfwGetKey(glfw_win, GLFW_KEY_DOWN) == GLFW_PRESS)  sphere_center.y -= spd;
      if (glfwGetKey(glfw_win, GLFW_KEY_PAGE_UP) == GLFW_PRESS)   sphere_center.z += spd;
      if (glfwGetKey(glfw_win, GLFW_KEY_PAGE_DOWN) == GLFW_PRESS) sphere_center.z -= spd;
      if (glfwGetKey(glfw_win, GLFW_KEY_EQUAL) == GLFW_PRESS)     sphere_radius += spd;
      if (glfwGetKey(glfw_win, GLFW_KEY_MINUS) == GLFW_PRESS)     sphere_radius = std::max(1.0f, sphere_radius - spd);
    }
    if (vox_mode == MODE_FILL_BOX) {
      if (glfwGetKey(glfw_win, GLFW_KEY_LEFT) == GLFW_PRESS)  box_center.x -= spd;
      if (glfwGetKey(glfw_win, GLFW_KEY_RIGHT) == GLFW_PRESS) box_center.x += spd;
      if (glfwGetKey(glfw_win, GLFW_KEY_UP) == GLFW_PRESS)    box_center.y += spd;
      if (glfwGetKey(glfw_win, GLFW_KEY_DOWN) == GLFW_PRESS)  box_center.y -= spd;
      if (glfwGetKey(glfw_win, GLFW_KEY_PAGE_UP) == GLFW_PRESS)   box_center.z += spd;
      if (glfwGetKey(glfw_win, GLFW_KEY_PAGE_DOWN) == GLFW_PRESS) box_center.z -= spd;
      if (glfwGetKey(glfw_win, GLFW_KEY_EQUAL) == GLFW_PRESS)     box_half += glm::vec3(spd);
      if (glfwGetKey(glfw_win, GLFW_KEY_MINUS) == GLFW_PRESS)     box_half = glm::max(box_half - glm::vec3(spd), glm::vec3(1.0f));
    }

    if (glfwGetKey(glfw_win, GLFW_KEY_LEFT_BRACKET) == GLFW_PRESS) noise_seed--;
    if (glfwGetKey(glfw_win, GLFW_KEY_RIGHT_BRACKET) == GLFW_PRESS) noise_seed++;

    auto fb = window->framebuffer_size();
    VkExtent2D want{ static_cast<uint32_t>(fb.first), static_cast<uint32_t>(fb.second) };
    if (want.width == 0 || want.height == 0) { std::this_thread::sleep_for(10ms); continue; }

    if (want.width != last.width || want.height != last.height) {
      vkDeviceWaitIdle(device.device());
      destroy_swapchain_stack();
      create_swapchain_stack(want.width, want.height);
      last = swapchain->extent();
      std::this_thread::sleep_for(1ms);
      continue;
    }

    float rwf = static_cast<float>(swapchain->extent().width) * kRenderScale;
    float rhf = static_cast<float>(swapchain->extent().height) * kRenderScale;
    float jx = halton(jitter_index & 1023u, 2) - 0.5f;
    float jy = halton(jitter_index & 1023u, 3) - 0.5f;
    jitter_index++;
    glm::mat4 view = glm::lookAt(cam.position, cam.position + cam.forward(), {0.0f,1.0f,0.0f});
    glm::mat4 proj = glm::perspective(glm::radians(90.0f), rwf / rhf, 0.1f, 100.0f);
    proj[2][0] += jx * 2.0f / rwf;
    proj[2][1] += jy * 2.0f / rhf;
    glm::mat4 view_proj = proj * view;

    CameraUBO ubo{};
    ubo.inv_view_proj = glm::inverse(view_proj);
    ubo.render_resolution = {rwf, rhf};
    ubo.output_resolution = { static_cast<float>(swapchain->extent().width),
                              static_cast<float>(swapchain->extent().height) };
    ubo.time = total_time;
    ubo.debug_normals = (glfwGetKey(glfw_win, GLFW_KEY_N) == GLFW_PRESS) ? 1.0f : 0.0f;
    ubo.debug_level   = (glfwGetKey(glfw_win, GLFW_KEY_L) == GLFW_PRESS) ? 1.0f : 0.0f;
    ubo.debug_steps   = (glfwGetKey(glfw_win, GLFW_KEY_H) == GLFW_PRESS) ? 1.0f : 0.0f;
    upload_buffer(allocator.raw(), device.device(), device.graphics_family(),
                  device.graphics_queue(), cam_buf, &ubo, sizeof(ubo));

    VoxParams vparams{};
    vparams.dim = {static_cast<int>(N), static_cast<int>(N), static_cast<int>(N)};
    vparams.frame = frame_counter++;
    vparams.volMin = {0.0f, 0.0f, 0.0f};
    vparams.volMax = {static_cast<float>(N), static_cast<float>(N), static_cast<float>(N)};
    vparams.boxCenter = box_center;
    vparams.boxHalf   = box_half;
    vparams.sphereCenter = sphere_center;
    vparams.sphereRadius = sphere_radius;
    vparams.mode = vox_mode;
    vparams.op = vox_op;
    vparams.noiseSeed = noise_seed;
    vparams.material = 1;
    vparams.regionMin = {0,0,0};
    vparams.regionMax = {static_cast<int>(N), static_cast<int>(N), static_cast<int>(N)};
    vparams.terrainFreq = 0.05f;
    upload_buffer(allocator.raw(), device.device(), device.graphics_family(),
                  device.graphics_queue(), vox_params_buf, &vparams, sizeof(vparams));

    ctx.dispatch_dim = {static_cast<uint32_t>(vparams.regionMax.x - vparams.regionMin.x),
                        static_cast<uint32_t>(vparams.regionMax.y - vparams.regionMin.y),
                        static_cast<uint32_t>(vparams.regionMax.z - vparams.regionMin.z)};

    commands->acquire_record_present(
      swapchain->vk(),
      const_cast<VkImage*>(swapchain->images().data()),
      const_cast<VkImageView*>(swapchain->image_views().data()),
      swapchain->image_format(), swapchain->extent(),
      device.graphics_queue(), device.present_queue(),
      &record_present, &ctx);
    vkQueueWaitIdle(device.graphics_queue());
    void* mapped = nullptr;
    vmaMapMemory(allocator.raw(), steps_buf.allocation, &mapped);
    uint32_t* stepData = static_cast<uint32_t*>(mapped);
    uint64_t sum = 0; uint32_t maxv = 0;
    uint32_t cells = ctx.steps_dim.width * ctx.steps_dim.height;
    for(uint32_t i=0;i<cells;i++){ sum += stepData[i]; if(stepData[i] > maxv) maxv = stepData[i]; }
    vmaUnmapMemory(allocator.raw(), steps_buf.allocation);
    float avg_steps = static_cast<float>(sum) /
                      (static_cast<float>(swapchain->extent().width) * static_cast<float>(swapchain->extent().height));
    float max_steps = static_cast<float>(maxv) / static_cast<float>(kStepsDown * kStepsDown);

    void* mapped_s = nullptr;
    vmaMapMemory(allocator.raw(), shadow_steps_buf.allocation, &mapped_s);
    uint32_t* stepDataS = static_cast<uint32_t*>(mapped_s);
    uint64_t sum_s = 0; uint32_t maxv_s = 0;
    uint32_t cells_s = ctx.shadow_steps_dim.width * ctx.shadow_steps_dim.height;
    for(uint32_t i=0;i<cells_s;i++){ sum_s += stepDataS[i]; if(stepDataS[i] > maxv_s) maxv_s = stepDataS[i]; }
    vmaUnmapMemory(allocator.raw(), shadow_steps_buf.allocation);
    float avg_steps_s = static_cast<float>(sum_s) /
                        (static_cast<float>(swapchain->extent().width) * static_cast<float>(swapchain->extent().height));
    float max_steps_s = static_cast<float>(maxv_s) / static_cast<float>(kStepsDown * kStepsDown);
    spdlog::info("avg_steps {:.2f} max_steps {:.2f} shadow_avg {:.2f} shadow_max {:.2f}",
                 avg_steps, max_steps, avg_steps_s, max_steps_s);

    std::this_thread::sleep_for(1ms);
  }

  vkDeviceWaitIdle(device.device());

  destroy_swapchain_stack();
  ray_pipeline.reset();
  present_pipeline.reset();
  if (dpool) vkDestroyDescriptorPool(device.device(), dpool, nullptr);
  if (comp_set) vkFreeDescriptorSets(device.device(), comp_pool, 1, &comp_set);
  if (comp_pool) vkDestroyDescriptorPool(device.device(), comp_pool, nullptr);
  if (comp_pipeline) vkDestroyPipeline(device.device(), comp_pipeline, nullptr);
  if (comp_layout) vkDestroyPipelineLayout(device.device(), comp_layout, nullptr);
  if (comp_dsl) vkDestroyDescriptorSetLayout(device.device(), comp_dsl, nullptr);
  if (comp_l1_set) vkFreeDescriptorSets(device.device(), comp_l1_pool, 1, &comp_l1_set);
  if (comp_l1_pool) vkDestroyDescriptorPool(device.device(), comp_l1_pool, nullptr);
  if (comp_l1_pipeline) vkDestroyPipeline(device.device(), comp_l1_pipeline, nullptr);
  if (comp_l1_layout) vkDestroyPipelineLayout(device.device(), comp_l1_layout, nullptr);
  if (comp_l1_dsl) vkDestroyDescriptorSetLayout(device.device(), comp_l1_dsl, nullptr);
  if (occ_storage_view) vkDestroyImageView(device.device(), occ_storage_view, nullptr);
  if (occ_view) vkDestroyImageView(device.device(), occ_view, nullptr);
  if (occ_l1_storage_view) vkDestroyImageView(device.device(), occ_l1_storage_view, nullptr);
  if (occ_l1_view) vkDestroyImageView(device.device(), occ_l1_view, nullptr);
  if (mat_storage_view) vkDestroyImageView(device.device(), mat_storage_view, nullptr);
  if (mat_view) vkDestroyImageView(device.device(), mat_view, nullptr);
  destroy_image3d(allocator.raw(), occ_img);
  destroy_image3d(allocator.raw(), occ_l1_img);
  destroy_image3d(allocator.raw(), mat_img);
  destroy_buffer(allocator.raw(), cam_buf);
  destroy_buffer(allocator.raw(), vox_buf);
  destroy_buffer(allocator.raw(), vox_params_buf);
  vkDestroySampler(device.device(), nearest_sampler, nullptr);
  vkDestroySampler(device.device(), linear_sampler, nullptr);
  destroy_transfer_context();
  allocator.destroy();

  spdlog::info("Shutdown.");
  return 0;
}

} // namespace engine
