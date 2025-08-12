#include <engine/engine.hpp>
#include <engine/log.hpp>
#include <engine/config.hpp>
#include <engine/platform/window.hpp>
#include <engine/gfx/vulkan_instance.hpp>
#include <engine/gfx/vulkan_surface.hpp>
#include <engine/gfx/vulkan_device.hpp>
#include <engine/gfx/vulkan_swapchain.hpp>
#include <engine/gfx/vulkan_commands.hpp>
#include <engine/gfx/vulkan_pipeline.hpp>

#include <spdlog/spdlog.h>
#include <thread>
#include <chrono>
#include <filesystem>

#ifdef _WIN32
  #include <windows.h>
#endif

namespace engine {

// ---- helpers ----
static std::filesystem::path exe_dir() {
#ifdef _WIN32
  char buf[MAX_PATH]{};
  DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
  return std::filesystem::path(std::string(buf, buf + n)).parent_path();
#else
  return std::filesystem::current_path();
#endif
}

struct DrawCtx { TrianglePipeline* pipe; };

// Record one triangle via dynamic rendering (viewport Y-flip)
static void record_triangle(VkCommandBuffer cmd, VkImage /*img*/, VkImageView view, VkFormat /*fmt*/, VkExtent2D extent, void* user) {
  auto* ctx = static_cast<DrawCtx*>(user);

  VkRenderingAttachmentInfo color{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
  color.imageView   = view;
  color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  color.loadOp  = VK_ATTACHMENT_LOAD_OP_CLEAR;
  color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  VkClearValue clear; clear.color = { {0.05f, 0.05f, 0.08f, 1.0f} };
  color.clearValue = clear;

  VkRenderingInfo ri{ VK_STRUCTURE_TYPE_RENDERING_INFO };
  ri.renderArea.offset = {0, 0};
  ri.renderArea.extent = extent;
  ri.layerCount = 1;
  ri.colorAttachmentCount = 1;
  ri.pColorAttachments = &color;

  vkCmdBeginRendering(cmd, &ri);

  // Flip Y so “up” is up on screen (Vulkan’s NDC is inverted vs GL)
  VkViewport vp{};
  vp.x = 0.0f;
  vp.y = static_cast<float>(extent.height);
  vp.width  = static_cast<float>(extent.width);
  vp.height = -static_cast<float>(extent.height); // negative height = Y flip
  vp.minDepth = 0.0f; vp.maxDepth = 1.0f;
  vkCmdSetViewport(cmd, 0, 1, &vp);

  VkRect2D sc{ {0,0}, extent };
  vkCmdSetScissor(cmd, 0, 1, &sc);

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, ctx->pipe->pipeline());

  float aspect = extent.width / (extent.height > 0 ? float(extent.height) : 1.0f);
  vkCmdPushConstants(cmd, ctx->pipe->layout(), VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(float), &aspect);

  vkCmdDraw(cmd, 3, 1, 0, 0);
  vkCmdEndRendering(cmd);
}

int run() {
  init_logging();
  log_boot_banner("engine");

  // Window
  WindowDesc wdesc{};
  wdesc.title = "vk_engine Step 8 (triangle)";
  auto window = create_window(wdesc);
  auto fb = window->framebuffer_size();

  // Instance (+validation)
  VulkanInstanceCreateInfo ici{};
  ici.enable_validation = cfg::kValidation;
  for (auto* e : platform_required_instance_extensions()) ici.extra_extensions.push_back(e);
  VulkanInstance instance{ici};

  // Surface
  VulkanSurface surface{instance.vk(), window->native_handle()};

  // Device
  VulkanDeviceCreateInfo dci{};
  dci.instance = instance.vk();
  dci.surface = surface.vk();
  dci.enable_validation = cfg::kValidation;
  VulkanDevice device{dci};

  // Swapchain
  VulkanSwapchainCreateInfo sci{};
  sci.physical = device.physical();
  sci.device   = device.device();
  sci.surface  = surface.vk();
  sci.desired_width  = static_cast<uint32_t>(fb.first);
  sci.desired_height = static_cast<uint32_t>(fb.second);
  VulkanSwapchain swapchain{sci};

  // Absolute shader paths next to the exe
  const auto shader_dir = exe_dir() / "shaders";
  const auto vs_path = (shader_dir / "tri.vert.spv").string();
  const auto fs_path = (shader_dir / "tri.frag.spv").string();
  spdlog::info("[vk] Using shaders: {}", shader_dir.string());

  // Pipeline
  TrianglePipelineCreateInfo pci{};
  pci.device = device.device();
  pci.color_format = swapchain.image_format();
  pci.vs_spv = vs_path;
  pci.fs_spv = fs_path;
  TrianglePipeline pipeline{pci};

  // Commands
  VulkanCommandsCreateInfo cci{};
  cci.device = device.device();
  cci.graphics_family = device.graphics_family();
  cci.image_count = static_cast<uint32_t>(swapchain.image_views().size());
  VulkanCommands commands{cci};

  DrawCtx ctx{ &pipeline };

  using namespace std::chrono_literals;
  while (!window->should_close()) {
    window->poll_events();

    commands.acquire_record_present(
      swapchain.vk(),
      const_cast<VkImage*>(swapchain.images().data()),
      const_cast<VkImageView*>(swapchain.image_views().data()),
      swapchain.image_format(),
      swapchain.extent(),
      device.graphics_queue(),
      device.present_queue(),
      &record_triangle,
      &ctx);

    std::this_thread::sleep_for(1ms);
  }

  vkDeviceWaitIdle(device.device());
  spdlog::info("Shutdown.");
  return 0;
}

} // namespace engine
