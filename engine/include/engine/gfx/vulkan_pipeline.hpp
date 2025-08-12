#pragma once
#include <vulkan/vulkan.h>
#include <string>

namespace engine {

struct TrianglePipelineCreateInfo {
  VkDevice device = VK_NULL_HANDLE;
  VkFormat color_format = VK_FORMAT_UNDEFINED;
  std::string vs_spv;
  std::string fs_spv;
};

// Graphics pipeline: set=0,binding=0 combined image sampler.
// Vertex format: location0 = vec2 pos, location1 = vec2 uv.
class TrianglePipeline {
public:
  explicit TrianglePipeline(const TrianglePipelineCreateInfo& ci);
  ~TrianglePipeline();

  TrianglePipeline(const TrianglePipeline&) = delete;
  TrianglePipeline& operator=(const TrianglePipeline&) = delete;

  VkPipeline             pipeline()     const { return pipeline_; }
  VkPipelineLayout       layout()       const { return layout_; }
  VkFormat               color_format() const { return color_format_; }
  VkDescriptorSetLayout  dset_layout()  const { return dset_layout_; }

  VkShaderModule load_module(const std::string& path);

private:
  VkDevice                dev_ = VK_NULL_HANDLE;
  VkPipelineLayout        layout_ = VK_NULL_HANDLE;
  VkPipeline              pipeline_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout   dset_layout_ = VK_NULL_HANDLE;
  VkFormat                color_format_ = VK_FORMAT_UNDEFINED;
};

} // namespace engine
