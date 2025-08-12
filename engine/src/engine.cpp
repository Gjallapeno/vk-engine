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
#include <engine/gfx/vulkan_pipeline.hpp>
#include <engine/gfx/memory.hpp>

#include <spdlog/spdlog.h>
#include <thread>
#include <chrono>
#include <filesystem>
#include <memory>
#include <vector>

#ifdef _WIN32
  #include <windows.h>
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

struct DrawCtx {
  TrianglePipeline* pipe = nullptr;
  VkDescriptorSet   dset = VK_NULL_HANDLE;
  VkBuffer          vbo  = VK_NULL_HANDLE;
  VkBuffer          ibo  = VK_NULL_HANDLE;
  VkIndexType       index_type = VK_INDEX_TYPE_UINT16;
  const std::vector<VkImage>* swap_images = nullptr;
  std::vector<Image2D>        depth_images;
  std::vector<VkImageView>    depth_views;
  std::vector<bool>           depth_first_use;
};

static void record_textured(VkCommandBuffer cmd, VkImage swap_img, VkImageView view,
                            VkFormat, VkExtent2D extent, void* user) {
  auto* ctx = static_cast<DrawCtx*>(user);

  uint32_t idx = 0;
  if (ctx->swap_images) {
    for (uint32_t i = 0; i < ctx->swap_images->size(); ++i) {
      if ((*ctx->swap_images)[i] == swap_img) { idx = i; break; }
    }
  }

  VkImageView depth_view = ctx->depth_views[idx];
  if (ctx->depth_first_use[idx]) {
    VkImageMemoryBarrier to_depth{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    to_depth.srcAccessMask = 0;
    to_depth.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    to_depth.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    to_depth.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    to_depth.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_depth.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_depth.image = ctx->depth_images[idx].image;
    to_depth.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    to_depth.subresourceRange.baseMipLevel = 0;
    to_depth.subresourceRange.levelCount = 1;
    to_depth.subresourceRange.baseArrayLayer = 0;
    to_depth.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &to_depth);
    ctx->depth_first_use[idx] = false;
  }

  VkRenderingAttachmentInfo color{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
  color.imageView   = view;
  color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  color.loadOp  = VK_ATTACHMENT_LOAD_OP_CLEAR;
  color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  VkClearValue clear; clear.color = { {0.06f, 0.07f, 0.10f, 1.0f} };
  color.clearValue = clear;

  VkRenderingAttachmentInfo depth{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
  depth.imageView   = depth_view;
  depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
  depth.loadOp  = VK_ATTACHMENT_LOAD_OP_CLEAR;
  depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  VkClearValue dclear; dclear.depthStencil = {1.0f, 0};
  depth.clearValue = dclear;

  VkRenderingInfo ri{ VK_STRUCTURE_TYPE_RENDERING_INFO };
  ri.renderArea.offset = {0, 0};
  ri.renderArea.extent = extent;
  ri.layerCount = 1;
  ri.colorAttachmentCount = 1;
  ri.pColorAttachments = &color;
  ri.pDepthAttachment = &depth;

  vkCmdBeginRendering(cmd, &ri);

  // Flipped viewport (negative height to keep winding consistent)
  VkViewport vp{};
  vp.x = 0.0f;
  vp.y = static_cast<float>(extent.height);
  vp.width  = static_cast<float>(extent.width);
  vp.height = -static_cast<float>(extent.height);
  vp.minDepth = 0.0f; vp.maxDepth = 1.0f;
  vkCmdSetViewport(cmd, 0, 1, &vp);

  VkRect2D sc{ {0,0}, extent };
  vkCmdSetScissor(cmd, 0, 1, &sc);

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, ctx->pipe->pipeline());

  float aspect = static_cast<float>(extent.width) /
                 (extent.height > 0 ? static_cast<float>(extent.height) : 1.0f);
  vkCmdPushConstants(cmd, ctx->pipe->layout(), VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(float), &aspect);

  VkBuffer vbos[] = { ctx->vbo };
  VkDeviceSize offs[] = { 0 };
  vkCmdBindVertexBuffers(cmd, 0, 1, vbos, offs);
  vkCmdBindIndexBuffer(cmd, ctx->ibo, 0, ctx->index_type);

  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, ctx->pipe->layout(),
                          0, 1, &ctx->dset, 0, nullptr);

  vkCmdDrawIndexed(cmd, 6, 1, 0, 0, 0);
  vkCmdEndRendering(cmd);
}

int run() {
  init_logging();
  log_boot_banner("engine");

  WindowDesc wdesc{}; wdesc.title = "vk_engine Step 11 (textured quad)";
  auto window = create_window(wdesc);

  VulkanInstanceCreateInfo ici{}; ici.enable_validation = cfg::kValidation;
  for (auto* e : platform_required_instance_extensions()) ici.extra_extensions.push_back(e);
  VulkanInstance instance{ici};
  VulkanSurface surface{instance.vk(), window->native_handle()};
  VulkanDeviceCreateInfo dci{}; dci.instance = instance.vk(); dci.surface = surface.vk(); dci.enable_validation = cfg::kValidation;
  VulkanDevice device{dci};

  const auto shader_dir = exe_dir() / "shaders";
  const auto vs_path = (shader_dir / "tex.vert.spv").string();
  const auto fs_path = (shader_dir / "tex.frag.spv").string();
  spdlog::info("[vk] Using shaders: {}", shader_dir.string());

  GpuAllocator allocator; allocator.init(instance.vk(), device.physical(), device.device());

  // Rectangle geometry (CCW order)
  const float verts[] = {
    //   x,     y,     z,     u,   v
    -0.9f, -0.6f,  0.0f,  0.0f, 1.0f,  // 0: bottom-left
     0.9f, -0.6f,  0.0f,  1.0f, 1.0f,  // 1: bottom-right
     0.9f,  0.6f,  0.0f,  1.0f, 0.0f,  // 2: top-right
    -0.9f,  0.6f,  0.0f,  0.0f, 0.0f   // 3: top-left
  };
  const uint16_t indices[] = { 0, 1, 2, 2, 3, 0 };

  Buffer vbo = create_buffer(allocator.raw(), sizeof(verts), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
  upload_buffer(allocator.raw(), device.device(), device.graphics_family(), device.graphics_queue(),
                vbo, verts, sizeof(verts));
  Buffer ibo = create_buffer(allocator.raw(), sizeof(indices), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
  upload_buffer(allocator.raw(), device.device(), device.graphics_family(), device.graphics_queue(),
                ibo, indices, sizeof(indices));

  // Checkerboard
  const uint32_t TEX_W = 512, TEX_H = 512;
  std::vector<uint32_t> pixels(TEX_W * TEX_H);
  for (uint32_t y=0; y<TEX_H; ++y) {
    for (uint32_t x=0; x<TEX_W; ++x) {
      bool c = ((x/32) ^ (y/32)) & 1;
      uint8_t r = c ? 255 : 40, g = c ? 220 : 60, b = c ? 120 : 200;
      pixels[y*TEX_W + x] = (255u<<24) | (uint32_t(b)<<16) | (uint32_t(g)<<8) | uint32_t(r);
    }
  }

  Image2D img = create_image2d(allocator.raw(), TEX_W, TEX_H, VK_FORMAT_R8G8B8A8_UNORM,
                               VK_IMAGE_USAGE_SAMPLED_BIT);
  upload_image2d(allocator.raw(), device.device(), device.graphics_family(), device.graphics_queue(),
                 pixels.data(), pixels.size()*sizeof(uint32_t), img);

  // View + Sampler
  VkImageView view = VK_NULL_HANDLE;
  {
    VkImageViewCreateInfo vi{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vi.image = img.image; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = VK_FORMAT_R8G8B8A8_UNORM;
    vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vi.subresourceRange.levelCount = img.mip_levels; vi.subresourceRange.layerCount = 1;
    VK_CHECK(vkCreateImageView(device.device(), &vi, nullptr, &view));
  }
  VkSampler sampler = VK_NULL_HANDLE;
  {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(device.physical(), &props);
    VkPhysicalDeviceFeatures feats{};
    vkGetPhysicalDeviceFeatures(device.physical(), &feats);

    VkSamplerCreateInfo si{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    si.magFilter = VK_FILTER_LINEAR; si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.minLod = 0.0f; si.maxLod = static_cast<float>(img.mip_levels);
    if (feats.samplerAnisotropy) {
      si.anisotropyEnable = VK_TRUE;
      si.maxAnisotropy = props.limits.maxSamplerAnisotropy;
    } else {
      si.anisotropyEnable = VK_FALSE;
      si.maxAnisotropy = 1.0f;
    }
    VK_CHECK(vkCreateSampler(device.device(), &si, nullptr, &sampler));
  }

  std::unique_ptr<VulkanSwapchain>   swapchain;
  std::unique_ptr<VulkanCommands>    commands;
  std::unique_ptr<TrianglePipeline>  pipeline;

  VkDescriptorPool dpool = VK_NULL_HANDLE;
  VkDescriptorSet  dset  = VK_NULL_HANDLE;

  DrawCtx ctx{};

  // Descriptor pool is created once at startup so that descriptor sets can
  // be freed and re-allocated without recreating the pool on swapchain
  // recreation.
  {
    VkDescriptorPoolSize sizes{}; sizes.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; sizes.descriptorCount = 1;
    VkDescriptorPoolCreateInfo dpci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpci.maxSets = 1; dpci.poolSizeCount = 1; dpci.pPoolSizes = &sizes;
    VK_CHECK(vkCreateDescriptorPool(device.device(), &dpci, nullptr, &dpool));
  }

  auto destroy_swapchain_stack = [&]() {
    for (auto v : ctx.depth_views) vkDestroyImageView(device.device(), v, nullptr);
    ctx.depth_views.clear();
    for (auto& di : ctx.depth_images) destroy_image2d(allocator.raw(), di);
    ctx.depth_images.clear();
    ctx.depth_first_use.clear();
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

    if (!pipeline || pipeline->color_format() != swapchain->image_format()) {
      TrianglePipelineCreateInfo pci{};
      pci.device = device.device();
      pci.pipeline_cache = device.pipeline_cache();
      pci.color_format = swapchain->image_format();
      pci.depth_format = VK_FORMAT_D32_SFLOAT;
      pci.vs_spv = vs_path; pci.fs_spv = fs_path;
      pipeline = std::make_unique<TrianglePipeline>(pci);
    }

    if (dset) {
      vkFreeDescriptorSets(device.device(), dpool, 1, &dset);
      dset = VK_NULL_HANDLE;
    }

    VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    ai.descriptorPool = dpool;
    VkDescriptorSetLayout layout = pipeline->dset_layout();
    ai.descriptorSetCount = 1; ai.pSetLayouts = &layout;
    VK_CHECK(vkAllocateDescriptorSets(device.device(), &ai, &dset));

    VkDescriptorImageInfo ii{}; ii.sampler = sampler; ii.imageView = view;
    ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkWriteDescriptorSet write{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    write.dstSet = dset; write.dstBinding = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; write.descriptorCount = 1;
    write.pImageInfo = &ii;
    vkUpdateDescriptorSets(device.device(), 1, &write, 0, nullptr);

    ctx.swap_images = &swapchain->images();
    size_t count = swapchain->image_views().size();
    ctx.depth_images.resize(count);
    ctx.depth_views.resize(count);
    ctx.depth_first_use.assign(count, true);
    for (size_t i = 0; i < count; ++i) {
      ctx.depth_images[i] = create_image2d(allocator.raw(), swapchain->extent().width,
                                          swapchain->extent().height,
                                          VK_FORMAT_D32_SFLOAT,
                                          VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
      VkImageViewCreateInfo vi{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
      vi.image = ctx.depth_images[i].image;
      vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
      vi.format = VK_FORMAT_D32_SFLOAT;
      vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
      vi.subresourceRange.levelCount = 1;
      vi.subresourceRange.layerCount = 1;
      VK_CHECK(vkCreateImageView(device.device(), &vi, nullptr, &ctx.depth_views[i]));
    }
  };

  { auto fb = window->framebuffer_size();
    create_swapchain_stack(static_cast<uint32_t>(fb.first), static_cast<uint32_t>(fb.second)); }

  ctx.pipe = pipeline.get(); ctx.dset = dset; ctx.vbo = vbo.buffer; ctx.ibo = ibo.buffer;
  VkExtent2D last = swapchain->extent();
  using namespace std::chrono_literals;

  while (!window->should_close()) {
    window->poll_events();

    auto fb = window->framebuffer_size();
    VkExtent2D want{ static_cast<uint32_t>(fb.first), static_cast<uint32_t>(fb.second) };
    if (want.width == 0 || want.height == 0) { std::this_thread::sleep_for(10ms); continue; }

    if (want.width != last.width || want.height != last.height) {
      vkDeviceWaitIdle(device.device());
      destroy_swapchain_stack();
      create_swapchain_stack(want.width, want.height);
      ctx.pipe = pipeline.get(); ctx.dset = dset;
      last = swapchain->extent();
      std::this_thread::sleep_for(1ms);
      continue;
    }

    commands->acquire_record_present(
      swapchain->vk(),
      const_cast<VkImage*>(swapchain->images().data()),
      const_cast<VkImageView*>(swapchain->image_views().data()),
      swapchain->image_format(), swapchain->extent(),
      device.graphics_queue(), device.present_queue(),
      &record_textured, &ctx);

    std::this_thread::sleep_for(1ms);
  }

  vkDeviceWaitIdle(device.device());

  destroy_swapchain_stack();
  pipeline.reset();
  if (dset) vkFreeDescriptorSets(device.device(), dpool, 1, &dset);
  if (dpool) vkDestroyDescriptorPool(device.device(), dpool, nullptr);
  vkDestroySampler(device.device(), sampler, nullptr);
  vkDestroyImageView(device.device(), view, nullptr);
  destroy_image2d(allocator.raw(), img);
  destroy_buffer(allocator.raw(), ibo);
  destroy_buffer(allocator.raw(), vbo);
  destroy_transfer_context();
  allocator.destroy();

  spdlog::info("Shutdown.");
  return 0;
}

} // namespace engine
