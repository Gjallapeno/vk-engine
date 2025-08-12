#include <engine/gfx/utils.hpp>
#include <spdlog/spdlog.h>
#include <fstream>

namespace engine {

std::vector<char> load_spirv_file(const std::string& path) {
  std::ifstream f(path, std::ios::ate | std::ios::binary);
  if (!f) {
    spdlog::error("[vk] Failed to open SPIR-V: {}", path);
    std::abort();
  }
  size_t size = static_cast<size_t>(f.tellg());
  std::vector<char> data(size);
  f.seekg(0);
  f.read(data.data(), size);
  return data;
}

} // namespace engine

