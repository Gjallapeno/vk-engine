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
#include <cstdint>
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
  PresentPipeline* pipe = nullptr;
  VkDescriptorSet   dset = VK_NULL_HANDLE;
  VkPipeline        comp_pipe = VK_NULL_HANDLE;
  VkPipelineLayout  comp_layout = VK_NULL_HANDLE;
  VkDescriptorSet   comp_set = VK_NULL_HANDLE;
  VkImage           occ_image = VK_NULL_HANDLE;
  VkExtent3D        occ_dim{0,0,0};
  bool              first_frame = true;
};

struct CameraUBO {
  glm::mat4 inv_view_proj{1.0f};
  glm::vec2 resolution{0.0f};
  float     time = 0.0f;
  float     _pad = 0.0f; // std140 alignment
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
  glm::vec3 boxA{0.0f}; float pad2 = 0.0f;
  glm::vec3 boxB{0.0f}; float pad3 = 0.0f;
  glm::vec3 sphereCenter{0.0f}; float sphereRadius = 0.0f;
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

  VkImageMemoryBarrier pre{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
  pre.srcAccessMask = ctx->first_frame ? 0 : VK_ACCESS_SHADER_READ_BIT;
  pre.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  pre.oldLayout = ctx->first_frame ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  pre.newLayout = VK_IMAGE_LAYOUT_GENERAL;
  pre.image = ctx->occ_image;
  pre.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  pre.subresourceRange.levelCount = 1;
  pre.subresourceRange.layerCount = 1;
  VkPipelineStageFlags srcStage = ctx->first_frame ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                                                   : VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  vkCmdPipelineBarrier(cmd, srcStage, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       0, 0, nullptr, 0, nullptr, 1, &pre);

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->comp_pipe);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->comp_layout,
                          0, 1, &ctx->comp_set, 0, nullptr);
  const uint32_t gx = (ctx->occ_dim.width  + 7) / 8;
  const uint32_t gy = (ctx->occ_dim.height + 7) / 8;
  const uint32_t gz = (ctx->occ_dim.depth  + 7) / 8;
  vkCmdDispatch(cmd, gx, gy, gz);

  VkImageMemoryBarrier post{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
  post.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  post.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  post.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
  post.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  post.image = ctx->occ_image;
  post.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  post.subresourceRange.levelCount = 1;
  post.subresourceRange.layerCount = 1;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                       0, 0, nullptr, 0, nullptr, 1, &post);
  ctx->first_frame = false;

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
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, ctx->pipe->layout(),
                          0, 1, &ctx->dset, 0, nullptr);

  vkCmdDraw(cmd, 3, 1, 0, 0);
  vkCmdEndRendering(cmd);
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
  const auto fs_path = (shader_dir / "fs_present.frag.spv").string();
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
  VkImageView occ_view = VK_NULL_HANDLE;
  VkImageView occ_storage_view = VK_NULL_HANDLE;
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
  }

  // Upload static voxel bounds
  VoxelAABB vubo{};
  vubo.min = {0.0f,0.0f,0.0f};
  vubo.max = {static_cast<float>(N),static_cast<float>(N),static_cast<float>(N)};
  vubo.dim = {static_cast<int>(N),static_cast<int>(N),static_cast<int>(N)};
  upload_buffer(allocator.raw(), device.device(), device.graphics_family(),
                device.graphics_queue(), vox_buf, &vubo, sizeof(vubo));

  // Compute pipeline to generate voxel occupancy texture
  VkDescriptorSetLayout comp_dsl = VK_NULL_HANDLE;
  VkPipelineLayout      comp_layout = VK_NULL_HANDLE;
  VkPipeline            comp_pipeline = VK_NULL_HANDLE;
  VkDescriptorPool      comp_pool = VK_NULL_HANDLE;
  VkDescriptorSet       comp_set  = VK_NULL_HANDLE;
  {
    VkDescriptorSetLayoutBinding binds[2]{};
    binds[0].binding = 0; binds[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    binds[0].descriptorCount = 1; binds[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    binds[1].binding = 1; binds[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    binds[1].descriptorCount = 1; binds[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dlci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    dlci.bindingCount = 2; dlci.pBindings = binds;
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
    psizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; psizes[0].descriptorCount = 1;
    psizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; psizes[1].descriptorCount = 1;
    VkDescriptorPoolCreateInfo dpci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpci.maxSets = 1; dpci.poolSizeCount = 2; dpci.pPoolSizes = psizes;
    VK_CHECK(vkCreateDescriptorPool(device.device(), &dpci, nullptr, &comp_pool));

    VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    ai.descriptorPool = comp_pool; ai.descriptorSetCount = 1; ai.pSetLayouts = &comp_dsl;
    VK_CHECK(vkAllocateDescriptorSets(device.device(), &ai, &comp_set));

    VkDescriptorImageInfo img_info{};
    img_info.imageView = occ_storage_view;
    img_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorBufferInfo param_bi{};
    param_bi.buffer = vox_params_buf.buffer; param_bi.range = sizeof(VoxParams);
    VkWriteDescriptorSet ws[2]{};
    ws[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    ws[0].dstSet = comp_set; ws[0].dstBinding = 0;
    ws[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    ws[0].descriptorCount = 1; ws[0].pImageInfo = &img_info;
    ws[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    ws[1].dstSet = comp_set; ws[1].dstBinding = 1;
    ws[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    ws[1].descriptorCount = 1; ws[1].pBufferInfo = &param_bi;
    vkUpdateDescriptorSets(device.device(), 2, ws, 0, nullptr);
  }

  std::unique_ptr<VulkanSwapchain>   swapchain;
  std::unique_ptr<VulkanCommands>    commands;
  std::unique_ptr<PresentPipeline>   pipeline;

  VkDescriptorPool dpool = VK_NULL_HANDLE;
  VkDescriptorSet  dset  = VK_NULL_HANDLE;

  DrawCtx ctx{};

  // Descriptor pool for camera, voxel AABB and occupancy texture
  {
    VkDescriptorPoolSize sizes[2]{};
    sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; sizes[0].descriptorCount = 2;
    sizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; sizes[1].descriptorCount = 1;
    VkDescriptorPoolCreateInfo dpci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpci.maxSets = 1; dpci.poolSizeCount = 2; dpci.pPoolSizes = sizes;
    VK_CHECK(vkCreateDescriptorPool(device.device(), &dpci, nullptr, &dpool));
  }

  auto destroy_swapchain_stack = [&]() {
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
      PresentPipelineCreateInfo pci{};
      pci.device = device.device();
      pci.pipeline_cache = device.pipeline_cache();
      pci.color_format = swapchain->image_format();
      pci.vs_spv = vs_path; pci.fs_spv = fs_path;
      pipeline = std::make_unique<PresentPipeline>(pci);
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

    VkDescriptorBufferInfo cam_bi{}; cam_bi.buffer = cam_buf.buffer; cam_bi.range = sizeof(CameraUBO);
    VkDescriptorBufferInfo vox_bi{}; vox_bi.buffer = vox_buf.buffer; vox_bi.range = sizeof(VoxelAABB);
    VkDescriptorImageInfo occ_info{}; occ_info.sampler = nearest_sampler; occ_info.imageView = occ_view;
    occ_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet writes[3]{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = dset; writes[0].dstBinding = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[0].descriptorCount = 1; writes[0].pBufferInfo = &cam_bi;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = dset; writes[1].dstBinding = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[1].descriptorCount = 1; writes[1].pBufferInfo = &vox_bi;

    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = dset; writes[2].dstBinding = 2;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[2].descriptorCount = 1; writes[2].pImageInfo = &occ_info;

    vkUpdateDescriptorSets(device.device(), 3, writes, 0, nullptr);
  };

  { auto fb = window->framebuffer_size();
    create_swapchain_stack(static_cast<uint32_t>(fb.first), static_cast<uint32_t>(fb.second)); }

  ctx.pipe = pipeline.get(); ctx.dset = dset;
  ctx.comp_pipe = comp_pipeline; ctx.comp_layout = comp_layout; ctx.comp_set = comp_set;
  ctx.occ_image = occ_img.image; ctx.occ_dim = {N, N, N};
  ctx.first_frame = true;
  VkExtent2D last = swapchain->extent();
  float total_time = 0.0f;
  int   frame_counter = 0;
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
    glm::mat4 view_proj = cam.view_projection(
        static_cast<float>(want.width) / static_cast<float>(want.height),
        0.1f, 100.0f);

    CameraUBO ubo{};
    ubo.inv_view_proj = glm::inverse(view_proj);
    ubo.resolution = { static_cast<float>(swapchain->extent().width),
                       static_cast<float>(swapchain->extent().height) };
    ubo.time = total_time;
    upload_buffer(allocator.raw(), device.device(), device.graphics_family(),
                  device.graphics_queue(), cam_buf, &ubo, sizeof(ubo));

    VoxParams vparams{};
    vparams.dim = {static_cast<int>(N), static_cast<int>(N), static_cast<int>(N)};
    vparams.frame = frame_counter++;
    vparams.volMin = {0.0f, 0.0f, 0.0f};
    vparams.volMax = {static_cast<float>(N), static_cast<float>(N), static_cast<float>(N)};
    vparams.boxA = {20.0f,20.0f,20.0f};
    vparams.boxB = {100.0f,100.0f,100.0f};
    vparams.sphereCenter = {static_cast<float>(N)/2.0f, static_cast<float>(N)/2.0f, static_cast<float>(N)/2.0f};
    vparams.sphereRadius = 30.0f;
    upload_buffer(allocator.raw(), device.device(), device.graphics_family(),
                  device.graphics_queue(), vox_params_buf, &vparams, sizeof(vparams));

    commands->acquire_record_present(
      swapchain->vk(),
      const_cast<VkImage*>(swapchain->images().data()),
      const_cast<VkImageView*>(swapchain->image_views().data()),
      swapchain->image_format(), swapchain->extent(),
      device.graphics_queue(), device.present_queue(),
      &record_present, &ctx);

    std::this_thread::sleep_for(1ms);
  }

  vkDeviceWaitIdle(device.device());

  destroy_swapchain_stack();
  pipeline.reset();
  if (dset) vkFreeDescriptorSets(device.device(), dpool, 1, &dset);
  if (dpool) vkDestroyDescriptorPool(device.device(), dpool, nullptr);
  if (comp_set) vkFreeDescriptorSets(device.device(), comp_pool, 1, &comp_set);
  if (comp_pool) vkDestroyDescriptorPool(device.device(), comp_pool, nullptr);
  if (comp_pipeline) vkDestroyPipeline(device.device(), comp_pipeline, nullptr);
  if (comp_layout) vkDestroyPipelineLayout(device.device(), comp_layout, nullptr);
  if (comp_dsl) vkDestroyDescriptorSetLayout(device.device(), comp_dsl, nullptr);
  if (occ_storage_view) vkDestroyImageView(device.device(), occ_storage_view, nullptr);
  if (occ_view) vkDestroyImageView(device.device(), occ_view, nullptr);
  destroy_image3d(allocator.raw(), occ_img);
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
