#include "glfw_context.hpp"
#include <GLFW/glfw3.h>
#include <stdexcept>

namespace engine {

std::mutex GlfwContext::mtx_{};
uint32_t GlfwContext::ref_count_ = 0;

GlfwContext::GlfwContext() {
  std::lock_guard<std::mutex> lock(mtx_);
  if (ref_count_ == 0) {
    if (!glfwInit()) throw std::runtime_error("GLFW init failed");
  }
  ++ref_count_;
}

GlfwContext::~GlfwContext() {
  std::lock_guard<std::mutex> lock(mtx_);
  if (--ref_count_ == 0) glfwTerminate();
}

} // namespace engine

