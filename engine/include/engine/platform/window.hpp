#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace engine {

struct WindowDesc {
  uint32_t width  = 1280;
  uint32_t height = 720;
  std::string title = "vk_engine";
  bool resizable = true;
};

class IWindow {
public:
  virtual ~IWindow() = default;
  virtual void poll_events() = 0;
  virtual bool should_close() const = 0;
  virtual std::pair<int,int> framebuffer_size() const = 0;
  virtual void set_title(const std::string& title) = 0;
  virtual void* native_handle() const = 0; // opaque native pointer (GLFWwindow*)
};

// Factory
std::unique_ptr<IWindow> create_window(const WindowDesc& desc);

// Vulkan helpers (from the platform layer; e.g. GLFW)
std::vector<const char*> platform_required_instance_extensions();

} // namespace engine
