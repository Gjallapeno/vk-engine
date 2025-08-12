#include <engine/platform/window.hpp>
#include <spdlog/spdlog.h>
#include <GLFW/glfw3.h>
#include <stdexcept>
#include <vector>

namespace engine {

class GlfwWindow final : public IWindow {
public:
  explicit GlfwWindow(const WindowDesc& d) {
    if (!glfwInit()) throw std::runtime_error("GLFW init failed");
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, d.resizable ? GLFW_TRUE : GLFW_FALSE);
    handle_ = glfwCreateWindow((int)d.width, (int)d.height, d.title.c_str(), nullptr, nullptr);
    if (!handle_) { glfwTerminate(); throw std::runtime_error("GLFW window creation failed"); }

    glfwSetFramebufferSizeCallback(handle_, [](GLFWwindow*, int x, int y){
      spdlog::trace("[event] framebuffer resize: {}x{}", x, y);
    });
    glfwSetWindowFocusCallback(handle_, [](GLFWwindow*, int focused){
      spdlog::trace("[event] focus: {}", focused ? "in" : "out");
    });

    int major, minor, rev; glfwGetVersion(&major,&minor,&rev);
    spdlog::info("GLFW {}.{}.{} | Window {}x{} created", major, minor, rev, d.width, d.height);
  }

  ~GlfwWindow() override {
    if (handle_) {
      spdlog::info("Destroying window");
      glfwDestroyWindow(handle_);
      glfwTerminate();
    }
  }

  void poll_events() override { glfwPollEvents(); }
  bool should_close() const override { return glfwWindowShouldClose(handle_) == GLFW_TRUE; }
  std::pair<int,int> framebuffer_size() const override {
    int x=0,y=0; glfwGetFramebufferSize(handle_, &x, &y); return {x,y};
  }
  void set_title(const std::string& t) override { glfwSetWindowTitle(handle_, t.c_str()); }
  void* native_handle() const override { return handle_; }

private:
  GLFWwindow* handle_ = nullptr;
};

std::unique_ptr<IWindow> create_window(const WindowDesc& desc) {
  return std::make_unique<GlfwWindow>(desc);
}

std::vector<const char*> platform_required_instance_extensions() {
  uint32_t count = 0;
  const char** ext = glfwGetRequiredInstanceExtensions(&count);
  std::vector<const char*> out;
  out.reserve(count);
  for (uint32_t i = 0; i < count; ++i) out.push_back(ext[i]);
  return out;
}

} // namespace engine
