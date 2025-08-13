#pragma once

#include <filesystem>
#include <string>
#include <vulkan/vulkan.h>

namespace engine {

std::filesystem::path exe_dir();
VkShaderModule load_module(VkDevice dev, const std::string &path);

} // namespace engine

