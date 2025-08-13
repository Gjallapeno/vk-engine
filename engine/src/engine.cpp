#include <engine/camera.hpp>
#include <engine/config.hpp>
#include <engine/engine.hpp>
#include <engine/brick_manager.hpp>
#include <engine/render_loop.hpp>
#include <engine/resource_setup.hpp>
#include <engine/gfx/memory.hpp>
#include <engine/gfx/present_pipeline.hpp>
#include <engine/gfx/ray_pipeline.hpp>
#include <engine/gfx/vulkan_commands.hpp>
#include <engine/gfx/vulkan_device.hpp>
#include <engine/gfx/vulkan_instance.hpp>
#include <engine/gfx/vulkan_surface.hpp>
#include <engine/gfx/vulkan_swapchain.hpp>
#include <engine/gfx/unique_handles.hpp>
#include <engine/log.hpp>
#include <engine/platform/window.hpp>
#include <engine/vk_checks.hpp>

#include <GLFW/glfw3.h>
#include <algorithm>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <glm/gtc/matrix_inverse.hpp>
#include <memory>
#include <spdlog/spdlog.h>
#include <vector>

namespace engine {

static constexpr uint32_t kStepsDown = 4;
static constexpr float kRenderScale = 0.66f;


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

int run() {

  init_logging();
  log_boot_banner("engine");

  WindowDesc wdesc{};
  wdesc.title = "vk_engine fullscreen present";
  auto window = create_window(wdesc);

  Camera cam;
  GLFWwindow *glfw_win = static_cast<GLFWwindow *>(window->native_handle());
  glfwSetInputMode(glfw_win, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
  double last_x = 0.0, last_y = 0.0;
  glfwGetCursorPos(glfw_win, &last_x, &last_y);
  auto last_time = std::chrono::high_resolution_clock::now();

  VulkanInstanceCreateInfo ici{};
  ici.enable_validation = cfg::kValidation;
  for (auto *e : platform_required_instance_extensions())
    ici.extra_extensions.push_back(e);
  VulkanInstance instance{ici};
  VulkanSurface surface{instance.vk(), window->native_handle()};
  VulkanDeviceCreateInfo dci{};
  dci.instance = instance.vk();
  dci.surface = surface.vk();
  dci.enable_validation = cfg::kValidation;
  VulkanDevice device{dci};
  load_debug_label_functions(instance.vk(), device.device());

  const auto shader_dir = exe_dir() / "shaders";
  const auto vs_path = (shader_dir / "vs_fullscreen.vert.spv").string();
  const auto ray_fs_path = (shader_dir / "fs_raycast.frag.spv").string();
  const auto light_fs_path = (shader_dir / "fs_lighting.frag.spv").string();
  spdlog::info("[vk] Using shaders: {}", shader_dir.string());

  GpuAllocator allocator;
  allocator.init(instance.vk(), device.physical(), device.device());
  {
    TransferContext transfer{device.device(), device.graphics_family(),
                             allocator.raw()};

  // Samplers for sampled images
  UniqueSampler linear_sampler;
  UniqueSampler nearest_sampler;
  {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(device.physical(), &props);
    VkPhysicalDeviceFeatures feats{};
    vkGetPhysicalDeviceFeatures(device.physical(), &feats);

    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.addressModeU = si.addressModeV = si.addressModeW =
        VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.minLod = 0.0f;
    si.maxLod = 0.0f;

    // linear sampler for floating point images
    si.magFilter = VK_FILTER_LINEAR;
    si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    if (feats.samplerAnisotropy) {
      si.anisotropyEnable = VK_TRUE;
      si.maxAnisotropy = props.limits.maxSamplerAnisotropy;
    } else {
      si.anisotropyEnable = VK_FALSE;
      si.maxAnisotropy = 1.0f;
    }
    VK_CHECK(
        vkCreateSampler(device.device(), &si, nullptr, linear_sampler.init(device.device())));

    // nearest sampler for integer textures (e.g. occupancy grid)
    si.magFilter = VK_FILTER_NEAREST;
    si.minFilter = VK_FILTER_NEAREST;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.anisotropyEnable = VK_FALSE;
    si.maxAnisotropy = 1.0f;
    VK_CHECK(
        vkCreateSampler(device.device(), &si, nullptr, nearest_sampler.init(device.device())));
  }

  // Buffers and occupancy texture
  Buffer cam_buf = create_buffer(allocator.raw(), sizeof(CameraUBO),
                                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
  Buffer vox_buf = create_buffer(allocator.raw(), sizeof(VoxelAABB),
                                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
  Buffer vox_params_buf = create_buffer(allocator.raw(), sizeof(VoxParams),
                                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
  const uint32_t N = 128;
  const uint32_t N1 = N / 4;
  const uint32_t N2 = N / 32;
  BrickManager brick_mgr;
  Image3D occ_l1_img{};
  Image3D occ_l2_img{};
  UniqueImageView occ_l1_view;
  UniqueImageView occ_l1_storage_view;
  UniqueImageView occ_l2_view;
  UniqueImageView occ_l2_storage_view;
  // brick_mgr provides views for occupancy/material/brick index textures
  {
    brick_mgr.init(allocator.raw(), device.device(), 32, 64, {N2, N2, N2});

    VkImageViewCreateInfo ovi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    ovi.viewType = VK_IMAGE_VIEW_TYPE_3D;
    ovi.format = VK_FORMAT_R8_UINT;
    ovi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    ovi.subresourceRange.levelCount = 1;
    ovi.subresourceRange.layerCount = 1;

    occ_l1_img =
        create_image3d(allocator.raw(), N1, N1, N1, VK_FORMAT_R8_UINT,
                       VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    ovi.image = occ_l1_img.image;
    VK_CHECK(vkCreateImageView(device.device(), &ovi, nullptr,
                               occ_l1_view.init(device.device())));
    VK_CHECK(vkCreateImageView(device.device(), &ovi, nullptr,
                               occ_l1_storage_view.init(device.device())));

    occ_l2_img =
        create_image3d(allocator.raw(), N2, N2, N2, VK_FORMAT_R8_UINT,
                       VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    ovi.image = occ_l2_img.image;
    ovi.format = VK_FORMAT_R8_UINT;
    VK_CHECK(vkCreateImageView(device.device(), &ovi, nullptr,
                               occ_l2_view.init(device.device())));
    VK_CHECK(vkCreateImageView(device.device(), &ovi, nullptr,
                               occ_l2_storage_view.init(device.device())));
  }

  // Upload static voxel bounds
  VoxelAABB vubo{};
  vubo.min = {0.0f, 0.0f, 0.0f};
  vubo.max = {static_cast<float>(N), static_cast<float>(N),
              static_cast<float>(N)};
  vubo.dim = {static_cast<int>(N), static_cast<int>(N), static_cast<int>(N)};
  vubo.occL1Dim = {static_cast<int>(N1), static_cast<int>(N1), static_cast<int>(N1)};
  vubo.occL1CellSize =
      (vubo.max - vubo.min) / glm::vec3(vubo.occL1Dim);
  vubo.occL2Dim = {static_cast<int>(N2), static_cast<int>(N2), static_cast<int>(N2)};
  vubo.occL2CellSize =
      (vubo.max - vubo.min) / glm::vec3(vubo.occL2Dim);
  upload_buffer(allocator.raw(), transfer, device.graphics_queue(), vox_buf,
                &vubo, sizeof(vubo));

  // Compute pipeline to generate voxel occupancy and material textures
  UniqueDescriptorSetLayout comp_dsl;
  UniquePipelineLayout comp_layout;
  UniquePipeline comp_pipeline;
  UniqueDescriptorPool comp_pool;
  VkDescriptorSet comp_set = VK_NULL_HANDLE;
  UniqueDescriptorSetLayout comp_l1_dsl;
  UniquePipelineLayout comp_l1_layout;
  UniquePipeline comp_l1_pipeline;
  UniqueDescriptorPool comp_l1_pool;
  DrawCtx ctx{};
  {
    VkDescriptorSetLayoutBinding binds[3]{};
    binds[0].binding = 0;
    binds[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    binds[0].descriptorCount = 1;
    binds[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    binds[1].binding = 1;
    binds[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    binds[1].descriptorCount = 1;
    binds[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    binds[2].binding = 2;
    binds[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    binds[2].descriptorCount = 1;
    binds[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dlci{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dlci.bindingCount = 3;
    dlci.pBindings = binds;
    VK_CHECK(vkCreateDescriptorSetLayout(device.device(), &dlci, nullptr,
                                         comp_dsl.init(device.device())));

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = sizeof(BrickParams);

    VkPipelineLayoutCreateInfo plci{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    VkDescriptorSetLayout comp_dsl_handle = comp_dsl.get();
    plci.pSetLayouts = &comp_dsl_handle;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    VK_CHECK(vkCreatePipelineLayout(device.device(), &plci, nullptr,
                                    comp_layout.init(device.device())));

    const auto cs_path = (shader_dir / "procgen_voxels.comp.spv").string();
    VkShaderModule cs = load_module(device.device(), cs_path);
    VkPipelineShaderStageCreateInfo stage{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = cs;
    stage.pName = "main";
    VkComputePipelineCreateInfo cpci{
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = stage;
    cpci.layout = comp_layout.get();
    VK_CHECK(vkCreateComputePipelines(device.device(), device.pipeline_cache(),
                                      1, &cpci, nullptr,
                                      comp_pipeline.init(device.device())));
    vkDestroyShaderModule(device.device(), cs, nullptr);

    VkDescriptorPoolSize psizes[2]{};
    psizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    psizes[0].descriptorCount = 2;
    psizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    psizes[1].descriptorCount = 1;
    VkDescriptorPoolCreateInfo dpci{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpci.maxSets = 1;
    dpci.poolSizeCount = 2;
    dpci.pPoolSizes = psizes;
    VK_CHECK(vkCreateDescriptorPool(device.device(), &dpci, nullptr,
                                    comp_pool.init(device.device())));

    VkDescriptorSetAllocateInfo ai{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = comp_pool.get();
    ai.descriptorSetCount = 1;
    comp_dsl_handle = comp_dsl.get();
    ai.pSetLayouts = &comp_dsl_handle;
    VK_CHECK(vkAllocateDescriptorSets(device.device(), &ai, &comp_set));

    VkDescriptorImageInfo occ_info{};
    occ_info.imageView = brick_mgr.occ_storage_view.get();
    occ_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorImageInfo mat_info{};
    mat_info.imageView = brick_mgr.mat_storage_view.get();
    mat_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorBufferInfo param_bi{};
    param_bi.buffer = vox_params_buf.buffer;
    param_bi.range = sizeof(VoxParams);
    VkWriteDescriptorSet ws[3]{};
    ws[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    ws[0].dstSet = comp_set;
    ws[0].dstBinding = 0;
    ws[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    ws[0].descriptorCount = 1;
    ws[0].pImageInfo = &occ_info;
    ws[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    ws[1].dstSet = comp_set;
    ws[1].dstBinding = 1;
    ws[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    ws[1].descriptorCount = 1;
    ws[1].pImageInfo = &mat_info;
    ws[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    ws[2].dstSet = comp_set;
    ws[2].dstBinding = 2;
    ws[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    ws[2].descriptorCount = 1;
    ws[2].pBufferInfo = &param_bi;
    vkUpdateDescriptorSets(device.device(), 3, ws, 0, nullptr);
  }

  // Compute pipeline to build coarse L1 occupancy from L0
  {
    VkDescriptorSetLayoutBinding binds[2]{};
    binds[0].binding = 0;
    binds[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    binds[0].descriptorCount = 1;
    binds[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    binds[1].binding = 1;
    binds[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    binds[1].descriptorCount = 1;
    binds[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dlci{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dlci.bindingCount = 2;
    dlci.pBindings = binds;
    VK_CHECK(vkCreateDescriptorSetLayout(device.device(), &dlci, nullptr,
                                         comp_l1_dsl.init(device.device())));

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = sizeof(BuildOccParams);
    VkPipelineLayoutCreateInfo plci{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    VkDescriptorSetLayout comp_l1_dsl_handle = comp_l1_dsl.get();
    plci.pSetLayouts = &comp_l1_dsl_handle;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    VK_CHECK(vkCreatePipelineLayout(device.device(), &plci, nullptr,
                                    comp_l1_layout.init(device.device())));

    const auto cs_path = (shader_dir / "build_occ_l1.comp.spv").string();
    VkShaderModule cs = load_module(device.device(), cs_path);
    VkPipelineShaderStageCreateInfo stage{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = cs;
    stage.pName = "main";
    VkComputePipelineCreateInfo cpci{
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = stage;
    cpci.layout = comp_l1_layout.get();
    VK_CHECK(vkCreateComputePipelines(device.device(), device.pipeline_cache(),
                                      1, &cpci, nullptr,
                                      comp_l1_pipeline.init(device.device())));
    vkDestroyShaderModule(device.device(), cs, nullptr);

    const uint32_t occ_levels = 3;
    VkDescriptorPoolSize psizes[2]{};
    psizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    psizes[0].descriptorCount = occ_levels - 1;
    psizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    psizes[1].descriptorCount = occ_levels - 1;
    VkDescriptorPoolCreateInfo dpci{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpci.maxSets = occ_levels - 1;
    dpci.poolSizeCount = 2;
    dpci.pPoolSizes = psizes;
    VK_CHECK(vkCreateDescriptorPool(device.device(), &dpci, nullptr,
                                    comp_l1_pool.init(device.device())));

    std::vector<VkDescriptorSet> comp_sets(occ_levels - 1);
    std::vector<VkDescriptorSetLayout> layouts(occ_levels - 1, comp_l1_dsl.get());
    VkDescriptorSetAllocateInfo ai{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = comp_l1_pool.get();
    ai.descriptorSetCount = occ_levels - 1;
    ai.pSetLayouts = layouts.data();
    VK_CHECK(vkAllocateDescriptorSets(device.device(), &ai, comp_sets.data()));

    std::vector<VkDescriptorImageInfo> src_infos(occ_levels - 1);
    std::vector<VkDescriptorImageInfo> dst_infos(occ_levels - 1);
    std::vector<VkWriteDescriptorSet> ws(2 * (occ_levels - 1));

    src_infos[0].sampler = nearest_sampler.get();
    src_infos[0].imageView = brick_mgr.occ_view.get();
    src_infos[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    dst_infos[0].imageView = occ_l1_storage_view.get();
    dst_infos[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    src_infos[1].sampler = nearest_sampler.get();
    src_infos[1].imageView = occ_l1_view.get();
    src_infos[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    dst_infos[1].imageView = occ_l2_storage_view.get();
    dst_infos[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    for (uint32_t i = 0; i < occ_levels - 1; ++i) {
      ws[2 * i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      ws[2 * i].dstSet = comp_sets[i];
      ws[2 * i].dstBinding = 0;
      ws[2 * i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      ws[2 * i].descriptorCount = 1;
      ws[2 * i].pImageInfo = &src_infos[i];

      ws[2 * i + 1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      ws[2 * i + 1].dstSet = comp_sets[i];
      ws[2 * i + 1].dstBinding = 1;
      ws[2 * i + 1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      ws[2 * i + 1].descriptorCount = 1;
      ws[2 * i + 1].pImageInfo = &dst_infos[i];
    }
    vkUpdateDescriptorSets(device.device(), static_cast<uint32_t>(ws.size()),
                           ws.data(), 0, nullptr);

    ctx.occ_view = brick_mgr.occ_view.get();
    ctx.occ_l1_view = occ_l1_view.get();
    ctx.occ_l1_storage_view = occ_l1_storage_view.get();
    ctx.occ_l2_view = occ_l2_view.get();
    ctx.occ_l2_storage_view = occ_l2_storage_view.get();
    ctx.occ_levels = occ_levels;
    ctx.comp_l1_sets = comp_sets;
  }

  std::unique_ptr<VulkanSwapchain> swapchain;
  std::unique_ptr<VulkanCommands> commands;
  std::unique_ptr<RayPipeline> ray_pipeline;
  std::unique_ptr<PresentPipeline> present_pipeline;

  UniqueDescriptorPool dpool;
  VkDescriptorSet ray_dset = VK_NULL_HANDLE;
  VkDescriptorSet light_dset = VK_NULL_HANDLE;

  // Descriptor pool for geometry and lighting passes
  {
    VkDescriptorPoolSize sizes[3]{};
    sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    sizes[0].descriptorCount = 3;
    sizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    // Ray pass now samples 5 images (L0 occ, material, L1 occ, L2 occ, brick ptr)
    // and the lighting pass samples 3 G-buffer images, so allocate 8 in total.
    sizes[1].descriptorCount = 8;
    sizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    sizes[2].descriptorCount = 1;
    VkDescriptorPoolCreateInfo dpci{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpci.maxSets = 2;
    dpci.poolSizeCount = 3;
    dpci.pPoolSizes = sizes;
    VK_CHECK(vkCreateDescriptorPool(device.device(), &dpci, nullptr,
                                    dpool.init(device.device())));
  }

  Image2D g_albedo_img{};
  Image2D g_normal_img{};
  Image2D g_depth_img{};
  Image2D steps_img{};
  UniqueImageView g_albedo_view;
  UniqueImageView g_normal_view;
  UniqueImageView g_depth_view;
  UniqueImageView steps_view;
  Buffer steps_buf{};

  auto destroy_swapchain_stack = [&]() {
    if (ray_dset) {
      vkFreeDescriptorSets(device.device(), dpool.get(), 1, &ray_dset);
      ray_dset = VK_NULL_HANDLE;
    }
    if (light_dset) {
      vkFreeDescriptorSets(device.device(), dpool.get(), 1, &light_dset);
      light_dset = VK_NULL_HANDLE;
    }
    g_albedo_view.reset();
    g_normal_view.reset();
    g_depth_view.reset();
    steps_view.reset();
    destroy_image2d(allocator.raw(), g_albedo_img);
    destroy_image2d(allocator.raw(), g_normal_img);
    destroy_image2d(allocator.raw(), g_depth_img);
    destroy_image2d(allocator.raw(), steps_img);
    destroy_buffer(allocator.raw(), steps_buf);
    commands.reset();
    swapchain.reset();
  };

  auto create_swapchain_stack = [&](uint32_t sw, uint32_t sh) {
    VulkanSwapchainCreateInfo sci{};
    sci.physical = device.physical();
    sci.device = device.device();
    sci.surface = surface.vk();
    sci.desired_width = sw;
    sci.desired_height = sh;
    swapchain = std::make_unique<VulkanSwapchain>(sci);

    VulkanCommandsCreateInfo cci{};
    cci.device = device.device();
    cci.graphics_family = device.graphics_family();
    cci.image_count = static_cast<uint32_t>(swapchain->image_views().size());
    commands = std::make_unique<VulkanCommands>(cci);

    if (!ray_pipeline) {
      RayPipelineCreateInfo rpci{};
      rpci.device = device.device();
      rpci.pipeline_cache = device.pipeline_cache();
      rpci.vs_spv = vs_path;
      rpci.fs_spv = ray_fs_path;
      ray_pipeline = std::make_unique<RayPipeline>(rpci);
    }
    if (!present_pipeline ||
        present_pipeline->color_format() != swapchain->image_format()) {
      PresentPipelineCreateInfo pci{};
      pci.device = device.device();
      pci.pipeline_cache = device.pipeline_cache();
      pci.color_format = swapchain->image_format();
      pci.vs_spv = vs_path;
      pci.fs_spv = light_fs_path;
      present_pipeline = std::make_unique<PresentPipeline>(pci);
    }

    if (ray_dset) {
      vkFreeDescriptorSets(device.device(), dpool.get(), 1, &ray_dset);
      ray_dset = VK_NULL_HANDLE;
    }
    if (light_dset) {
      vkFreeDescriptorSets(device.device(), dpool.get(), 1, &light_dset);
      light_dset = VK_NULL_HANDLE;
    }

    VkDescriptorSetAllocateInfo ai{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = dpool.get();
    VkDescriptorSetLayout layout = ray_pipeline->dset_layout();
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &layout;
    VK_CHECK(vkAllocateDescriptorSets(device.device(), &ai, &ray_dset));

    uint32_t rw = static_cast<uint32_t>(static_cast<float>(sw) * kRenderScale);
    uint32_t rh = static_cast<uint32_t>(static_cast<float>(sh) * kRenderScale);
    rw = std::max(1u, rw);
    rh = std::max(1u, rh);
    uint32_t steps_w = (rw + kStepsDown - 1) / kStepsDown;
    uint32_t steps_h = (rh + kStepsDown - 1) / kStepsDown;
    steps_img = create_image2d(
        allocator.raw(), steps_w, steps_h, VK_FORMAT_R32_UINT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
            VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    VkImageViewCreateInfo svi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    svi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    svi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    svi.subresourceRange.levelCount = 1;
    svi.subresourceRange.layerCount = 1;
    svi.image = steps_img.image;
    svi.format = VK_FORMAT_R32_UINT;
    VK_CHECK(vkCreateImageView(device.device(), &svi, nullptr,
                                steps_view.init(device.device())));
    steps_buf = create_host_buffer(allocator.raw(),
                                   steps_w * steps_h * sizeof(uint32_t),
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    VkDescriptorBufferInfo cam_bi{};
    cam_bi.buffer = cam_buf.buffer;
    cam_bi.range = sizeof(CameraUBO);
    VkDescriptorBufferInfo vox_bi{};
    vox_bi.buffer = vox_buf.buffer;
    vox_bi.range = sizeof(VoxelAABB);
    VkDescriptorImageInfo occ_info{};
    occ_info.sampler = nearest_sampler.get();
    occ_info.imageView = brick_mgr.occ_view.get();
    occ_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo mat_info{};
    mat_info.sampler = nearest_sampler.get();
    mat_info.imageView = brick_mgr.mat_view.get();
    mat_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo occ_l1_info{};
    occ_l1_info.sampler = nearest_sampler.get();
    occ_l1_info.imageView = occ_l1_view.get();
    occ_l1_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo occ_l2_info{};
    occ_l2_info.sampler = nearest_sampler.get();
    occ_l2_info.imageView = occ_l2_view.get();
    occ_l2_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo brick_info{};
    brick_info.sampler = nearest_sampler.get();
    brick_info.imageView = brick_mgr.index_view.get();
    brick_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo steps_info{};
    steps_info.imageView = steps_view.get();
    steps_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet rwrites[8]{};
    rwrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    rwrites[0].dstSet = ray_dset;
    rwrites[0].dstBinding = 0;
    rwrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    rwrites[0].descriptorCount = 1;
    rwrites[0].pBufferInfo = &cam_bi;
    rwrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    rwrites[1].dstSet = ray_dset;
    rwrites[1].dstBinding = 1;
    rwrites[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    rwrites[1].descriptorCount = 1;
    rwrites[1].pBufferInfo = &vox_bi;
    rwrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    rwrites[2].dstSet = ray_dset;
    rwrites[2].dstBinding = 2;
    rwrites[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    rwrites[2].descriptorCount = 1;
    rwrites[2].pImageInfo = &occ_info;
    rwrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    rwrites[3].dstSet = ray_dset;
    rwrites[3].dstBinding = 3;
    rwrites[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    rwrites[3].descriptorCount = 1;
    rwrites[3].pImageInfo = &mat_info;
    rwrites[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    rwrites[4].dstSet = ray_dset;
    rwrites[4].dstBinding = 4;
    rwrites[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    rwrites[4].descriptorCount = 1;
    rwrites[4].pImageInfo = &occ_l1_info;
    rwrites[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    rwrites[5].dstSet = ray_dset;
    rwrites[5].dstBinding = 5;
    rwrites[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    rwrites[5].descriptorCount = 1;
    rwrites[5].pImageInfo = &steps_info;
    rwrites[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    rwrites[6].dstSet = ray_dset;
    rwrites[6].dstBinding = 6;
    rwrites[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    rwrites[6].descriptorCount = 1;
    rwrites[6].pImageInfo = &occ_l2_info;
    rwrites[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    rwrites[7].dstSet = ray_dset;
    rwrites[7].dstBinding = 7;
    rwrites[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    rwrites[7].descriptorCount = 1;
    rwrites[7].pImageInfo = &brick_info;
    vkUpdateDescriptorSets(device.device(), 8, rwrites, 0, nullptr);

    g_albedo_img = create_image2d(
        allocator.raw(), rw, rh, VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    g_normal_img = create_image2d(
        allocator.raw(), rw, rh, VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    g_depth_img = create_image2d(allocator.raw(), rw, rh, VK_FORMAT_R32_SFLOAT,
                                 VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                     VK_IMAGE_USAGE_SAMPLED_BIT);

    VkImageViewCreateInfo iv{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    iv.viewType = VK_IMAGE_VIEW_TYPE_2D;
    iv.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    iv.subresourceRange.levelCount = 1;
    iv.subresourceRange.layerCount = 1;
    iv.image = g_albedo_img.image;
    iv.format = VK_FORMAT_R8G8B8A8_UNORM;
    VK_CHECK(vkCreateImageView(device.device(), &iv, nullptr,
                               g_albedo_view.init(device.device())));
    iv.image = g_normal_img.image;
    iv.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    VK_CHECK(vkCreateImageView(device.device(), &iv, nullptr,
                               g_normal_view.init(device.device())));
    iv.image = g_depth_img.image;
    iv.format = VK_FORMAT_R32_SFLOAT;
    VK_CHECK(vkCreateImageView(device.device(), &iv, nullptr,
                               g_depth_view.init(device.device())));

    VkDescriptorSetLayout layout2 = present_pipeline->dset_layout();
    ai.pSetLayouts = &layout2;
    VK_CHECK(vkAllocateDescriptorSets(device.device(), &ai, &light_dset));

    VkDescriptorImageInfo albedo_info{};
    albedo_info.sampler = linear_sampler.get();
    albedo_info.imageView = g_albedo_view.get();
    albedo_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo normal_info{};
    normal_info.sampler = linear_sampler.get();
    normal_info.imageView = g_normal_view.get();
    normal_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo depth_info{};
    depth_info.sampler = linear_sampler.get();
    depth_info.imageView = g_depth_view.get();
    depth_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet lwrites[4]{};
    lwrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    lwrites[0].dstSet = light_dset;
    lwrites[0].dstBinding = 0;
    lwrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    lwrites[0].descriptorCount = 1;
    lwrites[0].pBufferInfo = &cam_bi;
    lwrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    lwrites[1].dstSet = light_dset;
    lwrites[1].dstBinding = 1;
    lwrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    lwrites[1].descriptorCount = 1;
    lwrites[1].pImageInfo = &albedo_info;
    lwrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    lwrites[2].dstSet = light_dset;
    lwrites[2].dstBinding = 2;
    lwrites[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    lwrites[2].descriptorCount = 1;
    lwrites[2].pImageInfo = &normal_info;
    lwrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    lwrites[3].dstSet = light_dset;
    lwrites[3].dstBinding = 3;
    lwrites[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    lwrites[3].descriptorCount = 1;
    lwrites[3].pImageInfo = &depth_info;
    vkUpdateDescriptorSets(device.device(), 4, lwrites, 0, nullptr);

    ctx.ray_pipe = ray_pipeline.get();
    ctx.ray_dset = ray_dset;
    ctx.light_pipe = present_pipeline.get();
    ctx.light_dset = light_dset;
    ctx.g_albedo = g_albedo_img.image;
    ctx.g_normal = g_normal_img.image;
    ctx.g_depth = g_depth_img.image;
    ctx.g_albedo_view = g_albedo_view.get();
    ctx.g_normal_view = g_normal_view.get();
    ctx.g_depth_view = g_depth_view.get();
    ctx.steps_image = steps_img.image;
    ctx.steps_view = steps_view.get();
    ctx.steps_buffer = steps_buf.buffer;
    ctx.steps_dim = {steps_w, steps_h};
    ctx.ray_extent = {rw, rh};
    ctx.g_albedo_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    ctx.g_normal_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    ctx.g_depth_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    ctx.steps_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    ctx.first_frame = true;
  };

  {
    auto fb = window->framebuffer_size();
    create_swapchain_stack(static_cast<uint32_t>(fb.first),
                           static_cast<uint32_t>(fb.second));
  }

  ctx.comp_pipe = comp_pipeline.get();
  ctx.comp_layout = comp_layout.get();
  ctx.comp_set = comp_set;
  ctx.comp_l1_pipe = comp_l1_pipeline.get();
  ctx.comp_l1_layout = comp_l1_layout.get();
  ctx.occ_image = brick_mgr.occ_image();
  ctx.mat_image = brick_mgr.mat_image();
  ctx.occ_l1_image = occ_l1_img.image;
  ctx.occ_l2_image = occ_l2_img.image;
  ctx.occ_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  ctx.mat_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  ctx.occ_l1_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  ctx.occ_l2_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  ctx.brick_ptr_image = brick_mgr.index_image();
  ctx.occ_dim = {brick_mgr.brick_size, brick_mgr.brick_size,
                 brick_mgr.brick_size * brick_mgr.max_bricks};
  ctx.dispatch_dim = {brick_mgr.brick_size, brick_mgr.brick_size,
                      brick_mgr.brick_size};
  ctx.occ_l1_dim = {ctx.occ_dim.width / 4, ctx.occ_dim.height / 4,
                    ctx.occ_dim.depth / 4};
  ctx.dispatch_l1_dim = ctx.occ_l1_dim;
  ctx.occ_l2_dim = {ctx.occ_dim.width / 32, ctx.occ_dim.height / 32,
                    ctx.occ_dim.depth / 32};
  ctx.dispatch_l2_dim = ctx.occ_l2_dim;
  ctx.first_frame = true;
  VkExtent2D last = swapchain->extent();
  float total_time = 0.0f;
  int frame_counter = 0;
  uint32_t jitter_index = 0;
  using namespace std::chrono_literals;

  enum {
    MODE_CLEAR = 0,
    MODE_FILL_BOX = 1,
    MODE_FILL_SPHERE = 2,
    MODE_NOISE = 3,
    MODE_TERRAIN = 4
  };
  enum { OP_REPLACE = 0, OP_UNION = 1, OP_INTERSECTION = 2, OP_SUBTRACT = 3 };

  glm::vec3 sphere_center{static_cast<float>(N) / 2.0f,
                          static_cast<float>(N) / 2.0f,
                          static_cast<float>(N) / 2.0f};
  float sphere_radius = 30.0f;
  glm::vec3 box_center{60.0f, 60.0f, 60.0f};
  glm::vec3 box_half{40.0f, 40.0f, 40.0f};
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
    last_x = cx;
    last_y = cy;

    const float sens = 0.002f;
    cam.yaw += dx * sens;
    cam.pitch -= dy * sens;
    cam.pitch = std::clamp(cam.pitch, -glm::half_pi<float>() + 0.01f,
                           glm::half_pi<float>() - 0.01f);

    glm::vec3 move{0.0f};
    if (glfwGetKey(glfw_win, GLFW_KEY_W) == GLFW_PRESS)
      move += cam.forward();
    if (glfwGetKey(glfw_win, GLFW_KEY_S) == GLFW_PRESS)
      move -= cam.forward();
    if (glfwGetKey(glfw_win, GLFW_KEY_A) == GLFW_PRESS)
      move -= cam.right();
    if (glfwGetKey(glfw_win, GLFW_KEY_D) == GLFW_PRESS)
      move += cam.right();
    if (glfwGetKey(glfw_win, GLFW_KEY_SPACE) == GLFW_PRESS)
      move += glm::vec3(0.0f, 1.0f, 0.0f);
    if (glfwGetKey(glfw_win, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
      move -= glm::vec3(0.0f, 1.0f, 0.0f);
    if (glm::length(move) > 0.0f) {
      cam.position += glm::normalize(move) * (2.0f * dt);
    }

    // Voxel editing hotkeys
    if (glfwGetKey(glfw_win, GLFW_KEY_1) == GLFW_PRESS)
      vox_mode = MODE_CLEAR;
    if (glfwGetKey(glfw_win, GLFW_KEY_2) == GLFW_PRESS)
      vox_mode = MODE_FILL_BOX;
    if (glfwGetKey(glfw_win, GLFW_KEY_3) == GLFW_PRESS)
      vox_mode = MODE_FILL_SPHERE;
    if (glfwGetKey(glfw_win, GLFW_KEY_4) == GLFW_PRESS)
      vox_mode = MODE_NOISE;

    if (glfwGetKey(glfw_win, GLFW_KEY_5) == GLFW_PRESS)
      vox_op = OP_REPLACE;
    if (glfwGetKey(glfw_win, GLFW_KEY_6) == GLFW_PRESS)
      vox_op = OP_UNION;
    if (glfwGetKey(glfw_win, GLFW_KEY_7) == GLFW_PRESS)
      vox_op = OP_INTERSECTION;
    if (glfwGetKey(glfw_win, GLFW_KEY_8) == GLFW_PRESS)
      vox_op = OP_SUBTRACT;

    float spd = 30.0f * dt;
    if (vox_mode == MODE_FILL_SPHERE) {
      if (glfwGetKey(glfw_win, GLFW_KEY_LEFT) == GLFW_PRESS)
        sphere_center.x -= spd;
      if (glfwGetKey(glfw_win, GLFW_KEY_RIGHT) == GLFW_PRESS)
        sphere_center.x += spd;
      if (glfwGetKey(glfw_win, GLFW_KEY_UP) == GLFW_PRESS)
        sphere_center.y += spd;
      if (glfwGetKey(glfw_win, GLFW_KEY_DOWN) == GLFW_PRESS)
        sphere_center.y -= spd;
      if (glfwGetKey(glfw_win, GLFW_KEY_PAGE_UP) == GLFW_PRESS)
        sphere_center.z += spd;
      if (glfwGetKey(glfw_win, GLFW_KEY_PAGE_DOWN) == GLFW_PRESS)
        sphere_center.z -= spd;
      if (glfwGetKey(glfw_win, GLFW_KEY_EQUAL) == GLFW_PRESS)
        sphere_radius += spd;
      if (glfwGetKey(glfw_win, GLFW_KEY_MINUS) == GLFW_PRESS)
        sphere_radius = std::max(1.0f, sphere_radius - spd);
    }
    if (vox_mode == MODE_FILL_BOX) {
      if (glfwGetKey(glfw_win, GLFW_KEY_LEFT) == GLFW_PRESS)
        box_center.x -= spd;
      if (glfwGetKey(glfw_win, GLFW_KEY_RIGHT) == GLFW_PRESS)
        box_center.x += spd;
      if (glfwGetKey(glfw_win, GLFW_KEY_UP) == GLFW_PRESS)
        box_center.y += spd;
      if (glfwGetKey(glfw_win, GLFW_KEY_DOWN) == GLFW_PRESS)
        box_center.y -= spd;
      if (glfwGetKey(glfw_win, GLFW_KEY_PAGE_UP) == GLFW_PRESS)
        box_center.z += spd;
      if (glfwGetKey(glfw_win, GLFW_KEY_PAGE_DOWN) == GLFW_PRESS)
        box_center.z -= spd;
      if (glfwGetKey(glfw_win, GLFW_KEY_EQUAL) == GLFW_PRESS)
        box_half += glm::vec3(spd);
      if (glfwGetKey(glfw_win, GLFW_KEY_MINUS) == GLFW_PRESS)
        box_half = glm::max(box_half - glm::vec3(spd), glm::vec3(1.0f));
    }

    if (glfwGetKey(glfw_win, GLFW_KEY_LEFT_BRACKET) == GLFW_PRESS)
      noise_seed--;
    if (glfwGetKey(glfw_win, GLFW_KEY_RIGHT_BRACKET) == GLFW_PRESS)
      noise_seed++;

    auto fb = window->framebuffer_size();
    VkExtent2D want{static_cast<uint32_t>(fb.first),
                    static_cast<uint32_t>(fb.second)};
    if (want.width == 0 || want.height == 0) {
      std::this_thread::sleep_for(10ms);
      continue;
    }

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
    glm::mat4 view = glm::lookAt(cam.position, cam.position + cam.forward(),
                                 {0.0f, 1.0f, 0.0f});
    glm::mat4 proj =
        glm::perspective(glm::radians(90.0f), rwf / rhf, 0.1f, 100.0f);
    proj[2][0] += jx * 2.0f / rwf;
    proj[2][1] += jy * 2.0f / rhf;
    glm::mat4 view_proj = proj * view;

    CameraUBO ubo{};
    ubo.inv_view_proj = glm::inverse(view_proj);
    ubo.render_resolution = {rwf, rhf};
    ubo.output_resolution = {static_cast<float>(swapchain->extent().width),
                             static_cast<float>(swapchain->extent().height)};
    ubo.time = total_time;
    ubo.debug_normals =
        (glfwGetKey(glfw_win, GLFW_KEY_N) == GLFW_PRESS) ? 1.0f : 0.0f;
    ubo.debug_level =
        (glfwGetKey(glfw_win, GLFW_KEY_L) == GLFW_PRESS) ? 1.0f : 0.0f;
    ubo.debug_steps =
        (glfwGetKey(glfw_win, GLFW_KEY_H) == GLFW_PRESS) ? 1.0f : 0.0f;
    upload_buffer(allocator.raw(), transfer, device.graphics_queue(), cam_buf,
                  &ubo, sizeof(ubo));

    VoxParams vparams{};
    vparams.dim = {static_cast<int>(brick_mgr.brick_size),
                   static_cast<int>(brick_mgr.brick_size),
                   static_cast<int>(brick_mgr.brick_size)};
    vparams.frame = frame_counter++;
    vparams.volMin = {0.0f, 0.0f, 0.0f};
    vparams.volMax = {static_cast<float>(N), static_cast<float>(N),
                      static_cast<float>(N)};
    vparams.boxCenter = box_center;
    vparams.boxHalf = box_half;
    vparams.sphereCenter = sphere_center;
    vparams.sphereRadius = sphere_radius;
    vparams.mode = vox_mode;
    vparams.op = vox_op;
    vparams.noiseSeed = noise_seed;
    vparams.material = 1;
    vparams.terrainFreq = 0.05f;
    vparams.grassDensity = 0.5f;
    vparams.treeDensity = 0.05f;
    vparams.flowerDensity = 0.2f;

    BrickManager::BrickKey cam_key{
        static_cast<int>(std::floor(cam.position.x / brick_mgr.brick_size)),
        static_cast<int>(std::floor(cam.position.y / brick_mgr.brick_size)),
        static_cast<int>(std::floor(cam.position.z / brick_mgr.brick_size))};
    brick_mgr.acquire(cam_key, vparams.frame);
    brick_mgr.stream(cam.position, vparams.frame, 2);
    upload_buffer(allocator.raw(), transfer, device.graphics_queue(),
                  vox_params_buf, &vparams, sizeof(vparams));

    ctx.dispatch_dim = {brick_mgr.brick_size, brick_mgr.brick_size,
                        brick_mgr.brick_size};
    ctx.brick_params.clear();
    for (const auto &kv : brick_mgr.bricks) {
      BrickParams bp{};
      bp.brickCoord = {kv.first.x, kv.first.y, kv.first.z};
      bp.brickIndex = static_cast<int>(kv.second.index);
      ctx.brick_params.push_back(bp);
    }

    commands->acquire_record_present(
        swapchain->vk(), const_cast<VkImage *>(swapchain->images().data()),
        const_cast<VkImageView *>(swapchain->image_views().data()),
        swapchain->image_format(), swapchain->extent(), device.graphics_queue(),
        device.present_queue(), &record_present, &ctx);
    vkQueueWaitIdle(device.graphics_queue());
    void *mapped = nullptr;
    vmaMapMemory(allocator.raw(), steps_buf.allocation, &mapped);
    uint32_t *stepData = static_cast<uint32_t *>(mapped);
    uint64_t sum = 0;
    uint32_t maxv = 0;
    uint32_t cells = ctx.steps_dim.width * ctx.steps_dim.height;
    for (uint32_t i = 0; i < cells; i++) {
      sum += stepData[i];
      if (stepData[i] > maxv)
        maxv = stepData[i];
    }
    vmaUnmapMemory(allocator.raw(), steps_buf.allocation);
    float avg_steps = static_cast<float>(sum) /
                      (static_cast<float>(swapchain->extent().width) *
                       static_cast<float>(swapchain->extent().height));
    float max_steps =
        static_cast<float>(maxv) / static_cast<float>(kStepsDown * kStepsDown);
    spdlog::info("avg_steps {:.2f} max_steps {:.2f}", avg_steps, max_steps);

    std::this_thread::sleep_for(1ms);
  }

  vkDeviceWaitIdle(device.device());

  destroy_swapchain_stack();
  ray_pipeline.reset();
  present_pipeline.reset();

  occ_l1_storage_view.reset();
  occ_l1_view.reset();
  occ_l2_storage_view.reset();
  occ_l2_view.reset();

  brick_mgr.destroy();

  destroy_image3d(allocator.raw(), occ_l1_img);
  destroy_image3d(allocator.raw(), occ_l2_img);
  destroy_buffer(allocator.raw(), cam_buf);
  destroy_buffer(allocator.raw(), vox_buf);
  destroy_buffer(allocator.raw(), vox_params_buf);
    nearest_sampler.reset();
    linear_sampler.reset();
  }
  allocator.destroy();

  spdlog::info("Shutdown.");
  return 0;
}

} // namespace engine
