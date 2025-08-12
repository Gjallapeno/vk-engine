#pragma once
#include <vulkan/vulkan.h>
#include <string>

namespace engine {

struct PresentPipelineCreateInfo {
  VkDevice device = VK_NULL_HANDLE;
  VkPipelineCache pipeline_cache = VK_NULL_HANDLE;
  VkFormat color_format = VK_FORMAT_UNDEFINED;
  std::string vs_spv;
  std::string fs_spv;
};

// Graphics pipeline for fullscreen lighting of a G-buffer.
// Descriptor set layout:
//   set0,binding0 uniform buffer    (CameraUBO)
//   set0,binding1 combined sampler  (gAlbedoRough)
//   set0,binding2 combined sampler  (gNormal)
//   set0,binding3 combined sampler  (gDepth)
//   set0,binding4 uniform buffer    (VoxelAABB)
//   set0,binding5 combined sampler  (L0 occupancy texture)
//   set0,binding6 combined sampler  (L1 occupancy texture)
//   set0,binding7 storage image     (shadow step count image)
class PresentPipeline {
public:
  explicit PresentPipeline(const PresentPipelineCreateInfo& ci);
  ~PresentPipeline();

  PresentPipeline(const PresentPipeline&) = delete;
  PresentPipeline& operator=(const PresentPipeline&) = delete;

  VkPipeline             pipeline()     const { return pipeline_; }
  VkPipelineLayout       layout()       const { return layout_; }
  VkFormat               color_format() const { return color_format_; }
  VkDescriptorSetLayout  dset_layout()  const { return dset_layout_; }

  VkShaderModule load_module(const std::string& path);

private:
  VkDevice              dev_ = VK_NULL_HANDLE;
  VkPipelineLayout      layout_ = VK_NULL_HANDLE;
  VkPipeline            pipeline_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout dset_layout_ = VK_NULL_HANDLE;
  VkFormat              color_format_ = VK_FORMAT_UNDEFINED;
};

} // namespace engine
