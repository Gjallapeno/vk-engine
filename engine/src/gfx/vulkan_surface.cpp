#include <engine/gfx/vulkan_surface.hpp>
#include <engine/vk_checks.hpp>
#include <spdlog/spdlog.h>
#include <GLFW/glfw3.h>

namespace engine {

VulkanSurface::VulkanSurface(VkInstance instance, void* glfw_window)
  : instance_(instance) {
  GLFWwindow* w = static_cast<GLFWwindow*>(glfw_window);
  VK_CHECK(glfwCreateWindowSurface(instance_, w, nullptr, &surf_));
  spdlog::info("[vk] Surface created.");
}

VulkanSurface::~VulkanSurface() {
  if (surf_) vkDestroySurfaceKHR(instance_, surf_, nullptr);
}

} // namespace engine
