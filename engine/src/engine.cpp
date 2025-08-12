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

static std::vector<uint8_t> make_occ_grid(uint32_t N) {
  std::vector<uint8_t> occ(N*N*N, 0);
  auto fill = [&](uint32_t x0, uint32_t x1,
                  uint32_t y0, uint32_t y1,
                  uint32_t z0, uint32_t z1) {
    for(uint32_t z=z0; z<z1; ++z)
      for(uint32_t y=y0; y<y1; ++y)
        for(uint32_t x=x0; x<x1; ++x)
          occ[x + y*N + z*N*N] = 1;
  };
  uint32_t s = N/8;
  fill(N/4, N/4+s, N/4, N/4+s, N/4, N/4+s);
  fill(N/2, N/2+s, N/2, N/2+s, N/8, N/8+s);
  return occ;
}

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
  Image3D occ_img{};
  VkImageView occ_view = VK_NULL_HANDLE;
  {
    const uint32_t N = 128;
    auto occ = make_occ_grid(N);
    occ_img = create_image3d(allocator.raw(), N, N, N,
                             VK_FORMAT_R8_UINT,
                             VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    upload_image3d(allocator.raw(), device.device(), device.graphics_family(),
                   device.graphics_queue(), occ.data(), occ.size(), occ_img);
    VkImageViewCreateInfo ovi{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    ovi.image = occ_img.image; ovi.viewType = VK_IMAGE_VIEW_TYPE_3D;
    ovi.format = VK_FORMAT_R8_UINT;
    ovi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    ovi.subresourceRange.levelCount = 1;
    ovi.subresourceRange.layerCount = 1;
    VK_CHECK(vkCreateImageView(device.device(), &ovi, nullptr, &occ_view));
  }

  // Upload static voxel bounds
  VoxelAABB vubo{};
  vubo.min = {0.0f,0.0f,0.0f};
  vubo.max = {128.0f,128.0f,128.0f};
  vubo.dim = {128,128,128};
  upload_buffer(allocator.raw(), device.device(), device.graphics_family(),
                device.graphics_queue(), vox_buf, &vubo, sizeof(vubo));

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
  if (occ_view) vkDestroyImageView(device.device(), occ_view, nullptr);
  destroy_image3d(allocator.raw(), occ_img);
  destroy_buffer(allocator.raw(), cam_buf);
  destroy_buffer(allocator.raw(), vox_buf);
  vkDestroySampler(device.device(), nearest_sampler, nullptr);
  vkDestroySampler(device.device(), linear_sampler, nullptr);
  destroy_transfer_context();
  allocator.destroy();

  spdlog::info("Shutdown.");
  return 0;
}

} // namespace engine
