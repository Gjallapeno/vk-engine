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
#include <engine/camera.hpp>

#include <GLFW/glfw3.h>
#include <algorithm>

#include <spdlog/spdlog.h>
#include <thread>
#include <chrono>
#include <filesystem>
#include <memory>
#include <vector>
#include <fstream>
#include <glm/gtc/matrix_inverse.hpp>

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
  glm::mat4                  view_proj{1.0f};
  // compute resources
  VkPipeline       comp_pipe = VK_NULL_HANDLE;
  VkPipelineLayout comp_layout = VK_NULL_HANDLE;
  VkDescriptorSet  comp_dset = VK_NULL_HANDLE;
  Image2D          voxel_img;
  VkImageView      voxel_view = VK_NULL_HANDLE;
  bool             voxel_first_use = true;
};

struct CameraUBO {
  glm::mat4 inv_view_proj{1.0f};
  glm::vec2 resolution{0.0f};
  float     time = 0.0f;
  float     _pad = 0.0f; // std140 alignment
  glm::vec4 world_min{-1.0f,-1.0f,-1.0f,0.0f};
  glm::vec4 world_max{ 1.0f, 1.0f, 1.0f,0.0f};
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

static void record_textured(VkCommandBuffer cmd, VkImage swap_img, VkImageView view,
                            VkFormat, VkExtent2D extent, void* user) {
  auto* ctx = static_cast<DrawCtx*>(user);

  uint32_t idx = 0;
  if (ctx->swap_images) {
    for (uint32_t i = 0; i < ctx->swap_images->size(); ++i) {
      if ((*ctx->swap_images)[i] == swap_img) { idx = i; break; }
    }
  }

  // compute pass writes to voxel image
  VkImageMemoryBarrier to_comp{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
  to_comp.srcAccessMask = ctx->voxel_first_use ? 0 : VK_ACCESS_SHADER_READ_BIT;
  to_comp.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  to_comp.oldLayout = ctx->voxel_first_use ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  to_comp.newLayout = VK_IMAGE_LAYOUT_GENERAL;
  to_comp.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_comp.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_comp.image = ctx->voxel_img.image;
  to_comp.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  to_comp.subresourceRange.baseMipLevel = 0;
  to_comp.subresourceRange.levelCount = 1;
  to_comp.subresourceRange.baseArrayLayer = 0;
  to_comp.subresourceRange.layerCount = 1;
  vkCmdPipelineBarrier(cmd,
                       ctx->voxel_first_use ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       0, 0, nullptr, 0, nullptr, 1, &to_comp);

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->comp_pipe);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->comp_layout,
                          0, 1, &ctx->comp_dset, 0, nullptr);
  uint32_t gx = (ctx->voxel_img.width + 7) / 8;
  uint32_t gy = (ctx->voxel_img.height + 7) / 8;
  vkCmdDispatch(cmd, gx, gy, 1);

  VkImageMemoryBarrier to_sample{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
  to_sample.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  to_sample.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  to_sample.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
  to_sample.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  to_sample.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_sample.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_sample.image = ctx->voxel_img.image;
  to_sample.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  to_sample.subresourceRange.baseMipLevel = 0;
  to_sample.subresourceRange.levelCount = 1;
  to_sample.subresourceRange.baseArrayLayer = 0;
  to_sample.subresourceRange.layerCount = 1;
  vkCmdPipelineBarrier(cmd,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                       0, 0, nullptr, 0, nullptr, 1, &to_sample);
  ctx->voxel_first_use = false;

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

  vkCmdPushConstants(cmd, ctx->pipe->layout(), VK_SHADER_STAGE_VERTEX_BIT, 0,
                     sizeof(glm::mat4), &ctx->view_proj);

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

  // Sampler for presenting the compute output
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
    si.minLod = 0.0f; si.maxLod = 0.0f;
    if (feats.samplerAnisotropy) {
      si.anisotropyEnable = VK_TRUE;
      si.maxAnisotropy = props.limits.maxSamplerAnisotropy;
    } else {
      si.anisotropyEnable = VK_FALSE;
      si.maxAnisotropy = 1.0f;
    }
    VK_CHECK(vkCreateSampler(device.device(), &si, nullptr, &sampler));
  }

  // Compute pipeline and resources
  const auto cs_path = (shader_dir / "pp_raycast.comp.spv").string();
  VkDescriptorPool comp_pool = VK_NULL_HANDLE;
  VkDescriptorSet  comp_dset = VK_NULL_HANDLE;
  VkDescriptorSetLayout comp_dset_layout = VK_NULL_HANDLE;
  VkPipeline comp_pipe = VK_NULL_HANDLE;
  VkPipelineLayout comp_layout = VK_NULL_HANDLE;
  Buffer cam_buf = create_buffer(allocator.raw(), sizeof(CameraUBO),
                                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
  Image3D occ_img{};
  VkImageView occ_view = VK_NULL_HANDLE;
  {
    VkDescriptorPoolSize sizes[3]{};
    sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; sizes[0].descriptorCount = 1;
    sizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; sizes[1].descriptorCount = 1;
    sizes[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; sizes[2].descriptorCount = 1;
    VkDescriptorPoolCreateInfo dpci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpci.maxSets = 1; dpci.poolSizeCount = 3; dpci.pPoolSizes = sizes;
    VK_CHECK(vkCreateDescriptorPool(device.device(), &dpci, nullptr, &comp_pool));

    VkDescriptorSetLayoutBinding binds[3]{};
    binds[0].binding = 0; binds[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    binds[0].descriptorCount = 1; binds[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    binds[1].binding = 1; binds[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    binds[1].descriptorCount = 1; binds[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    binds[2].binding = 2; binds[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    binds[2].descriptorCount = 1; binds[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dlci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    dlci.bindingCount = 3; dlci.pBindings = binds;
    VK_CHECK(vkCreateDescriptorSetLayout(device.device(), &dlci, nullptr, &comp_dset_layout));

    VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1; plci.pSetLayouts = &comp_dset_layout;
    VK_CHECK(vkCreatePipelineLayout(device.device(), &plci, nullptr, &comp_layout));

    VkShaderModule cs = load_module(device.device(), cs_path);
    VkPipelineShaderStageCreateInfo stage{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; stage.module = cs; stage.pName = "main";
    VkComputePipelineCreateInfo cpci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpci.stage = stage; cpci.layout = comp_layout;
    VK_CHECK(vkCreateComputePipelines(device.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &comp_pipe));
    vkDestroyShaderModule(device.device(), cs, nullptr);

    VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    ai.descriptorPool = comp_pool; ai.descriptorSetCount = 1; ai.pSetLayouts = &comp_dset_layout;
    VK_CHECK(vkAllocateDescriptorSets(device.device(), &ai, &comp_dset));

    VkDescriptorBufferInfo bi{}; bi.buffer = cam_buf.buffer; bi.range = sizeof(CameraUBO);
    VkWriteDescriptorSet write{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    write.dstSet = comp_dset; write.dstBinding = 1; write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    write.descriptorCount = 1; write.pBufferInfo = &bi;
    vkUpdateDescriptorSets(device.device(), 1, &write, 0, nullptr);

    const uint32_t N = 128;
    std::vector<uint8_t> occ(N*N*N, 0);
    for(uint32_t z=N/2; z<N; ++z)
      for(uint32_t y=0; y<N; ++y)
        for(uint32_t x=0; x<N; ++x)
          occ[x + y*N + z*N*N] = 1;
    uint32_t c0 = N/4; uint32_t c1 = c0 + N/8;
    for(uint32_t z=c0; z<c1; ++z)
      for(uint32_t y=c0; y<c1; ++y)
        for(uint32_t x=c0; x<c1; ++x)
          occ[x + y*N + z*N*N] = 1;

    occ_img = create_image3d(allocator.raw(), N, N, N,
                             VK_FORMAT_R8_UINT,
                             VK_IMAGE_USAGE_SAMPLED_BIT);
    upload_image3d(allocator.raw(), device.device(), device.graphics_family(),
                   device.graphics_queue(), occ.data(), occ.size(), occ_img);
    VkImageViewCreateInfo ovi{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    ovi.image = occ_img.image; ovi.viewType = VK_IMAGE_VIEW_TYPE_3D;
    ovi.format = VK_FORMAT_R8_UINT;
    ovi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    ovi.subresourceRange.levelCount = 1;
    ovi.subresourceRange.layerCount = 1;
    VK_CHECK(vkCreateImageView(device.device(), &ovi, nullptr, &occ_view));

    VkDescriptorImageInfo oi{}; oi.sampler = sampler; oi.imageView = occ_view;
    oi.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkWriteDescriptorSet wocc{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    wocc.dstSet = comp_dset; wocc.dstBinding = 2;
    wocc.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    wocc.descriptorCount = 1; wocc.pImageInfo = &oi;
    vkUpdateDescriptorSets(device.device(), 1, &wocc, 0, nullptr);
  }

  std::unique_ptr<VulkanSwapchain>   swapchain;
  std::unique_ptr<VulkanCommands>    commands;
  std::unique_ptr<TrianglePipeline>  pipeline;

  VkDescriptorPool dpool = VK_NULL_HANDLE;
  VkDescriptorSet  dset  = VK_NULL_HANDLE;

  DrawCtx ctx{};
  ctx.comp_pipe = comp_pipe;
  ctx.comp_layout = comp_layout;
  ctx.comp_dset = comp_dset;

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
    if (ctx.voxel_view) vkDestroyImageView(device.device(), ctx.voxel_view, nullptr);
    ctx.voxel_view = VK_NULL_HANDLE;
    destroy_image2d(allocator.raw(), ctx.voxel_img);
    ctx.voxel_first_use = true;
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

    // compute target image matching swapchain size
    ctx.voxel_img = create_image2d(allocator.raw(), swapchain->extent().width,
                                   swapchain->extent().height,
                                   VK_FORMAT_R16G16B16A16_SFLOAT,
                                   VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    VkImageViewCreateInfo cvi{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    cvi.image = ctx.voxel_img.image; cvi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    cvi.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    cvi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    cvi.subresourceRange.levelCount = 1; cvi.subresourceRange.layerCount = 1;
    VK_CHECK(vkCreateImageView(device.device(), &cvi, nullptr, &ctx.voxel_view));
    ctx.voxel_first_use = true;

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

    VkDescriptorImageInfo ii{}; ii.sampler = sampler; ii.imageView = ctx.voxel_view;
    ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkWriteDescriptorSet write{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    write.dstSet = dset; write.dstBinding = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; write.descriptorCount = 1;
    write.pImageInfo = &ii;
    vkUpdateDescriptorSets(device.device(), 1, &write, 0, nullptr);

    VkDescriptorImageInfo si{}; si.imageView = ctx.voxel_view; si.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkWriteDescriptorSet wcomp{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    wcomp.dstSet = comp_dset; wcomp.dstBinding = 0;
    wcomp.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; wcomp.descriptorCount = 1;
    wcomp.pImageInfo = &si;
    vkUpdateDescriptorSets(device.device(), 1, &wcomp, 0, nullptr);

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
  float total_time = 0.0f;
  using namespace std::chrono_literals;

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
    ctx.view_proj = cam.view_projection(
        static_cast<float>(want.width) / static_cast<float>(want.height),
        0.1f, 100.0f);

    CameraUBO ubo{};
    ubo.inv_view_proj = glm::inverse(ctx.view_proj);
    ubo.resolution = { static_cast<float>(ctx.voxel_img.width), static_cast<float>(ctx.voxel_img.height) };
    ubo.time = total_time;
    upload_buffer(allocator.raw(), device.device(), device.graphics_family(),
                  device.graphics_queue(), cam_buf, &ubo, sizeof(ubo));

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
  if (comp_dset) vkFreeDescriptorSets(device.device(), comp_pool, 1, &comp_dset);
  if (comp_pool) vkDestroyDescriptorPool(device.device(), comp_pool, nullptr);
  if (comp_pipe) vkDestroyPipeline(device.device(), comp_pipe, nullptr);
  if (comp_layout) vkDestroyPipelineLayout(device.device(), comp_layout, nullptr);
  if (comp_dset_layout) vkDestroyDescriptorSetLayout(device.device(), comp_dset_layout, nullptr);
  if (occ_view) vkDestroyImageView(device.device(), occ_view, nullptr);
  destroy_image3d(allocator.raw(), occ_img);
  vkDestroySampler(device.device(), sampler, nullptr);
  destroy_buffer(allocator.raw(), cam_buf);
  destroy_buffer(allocator.raw(), ibo);
  destroy_buffer(allocator.raw(), vbo);
  destroy_transfer_context();
  allocator.destroy();

  spdlog::info("Shutdown.");
  return 0;
}

} // namespace engine
