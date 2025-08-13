#include <engine/resource_setup.hpp>
#include <engine/vk_checks.hpp>
#include <spdlog/spdlog.h>
#include <fstream>
#include <vector>
#include <cstdlib>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace engine {

std::filesystem::path exe_dir() {
#ifdef _WIN32
  char buf[MAX_PATH]{};
  DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
  return std::filesystem::path(std::string(buf, buf + n)).parent_path();
#else
  return std::filesystem::current_path();
#endif
}

VkShaderModule load_module(VkDevice dev, const std::string &path) {
  std::ifstream f(path, std::ios::ate | std::ios::binary);
  if (!f) {
    spdlog::error("[vk] Failed to open SPIR-V: {}", path);
    std::abort();
  }
  size_t size = static_cast<size_t>(f.tellg());
  std::vector<char> data(size);
  f.seekg(0);
  f.read(data.data(), size);
  VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  ci.codeSize = data.size();
  ci.pCode = reinterpret_cast<const uint32_t *>(data.data());
  VkShaderModule mod = VK_NULL_HANDLE;
  VK_CHECK(vkCreateShaderModule(dev, &ci, nullptr, &mod));
  return mod;
}

} // namespace engine

