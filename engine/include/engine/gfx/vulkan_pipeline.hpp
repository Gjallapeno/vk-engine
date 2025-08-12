#pragma once
#include <vulkan/vulkan.h>
#include <string>

namespace engine {

struct TrianglePipelineCreateInfo {
  VkDevice device = VK_NULL_HANDLE;
  VkFormat color_format = VK_FORMAT_B8G8R8A8_SRGB;
  std::string vs_spv;
  std::string fs_spv;
};

class TrianglePipeline {
public:
  explicit TrianglePipeline(const TrianglePipelineCreateInfo& ci);
  ~TrianglePipeline();

  TrianglePipeline(const TrianglePipeline&) = delete;
  TrianglePipeline& operator=(const TrianglePipeline&) = delete;

  VkPipelineLayout layout() const { return layout_; }
  VkPipeline pipeline() const { return pipeline_; }

private:
  VkShaderModule load_module(const std::string& path);

private:
  VkDevice dev_ = VK_NULL_HANDLE;
  VkPipelineLayout layout_ = VK_NULL_HANDLE;
  VkPipeline pipeline_ = VK_NULL_HANDLE;
};

} // namespace engine
