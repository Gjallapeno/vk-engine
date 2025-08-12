#pragma once
#include <vulkan/vulkan.h>
#include <string>

namespace engine {

struct RayPipelineCreateInfo {
  VkDevice device = VK_NULL_HANDLE;
  VkPipelineCache pipeline_cache = VK_NULL_HANDLE;
  std::string vs_spv;
  std::string fs_spv;
};

// Graphics pipeline for voxel raycasting writing to a G-buffer.
// Descriptor set layout:
//   set0,binding0 uniform buffer    (CameraUBO)
//   set0,binding1 uniform buffer    (VoxelAABB)
//   set0,binding2 storage buffer   (L0 occupancy bitfield)
//   set0,binding3 storage buffer   (material bitfield)
//   set0,binding4 combined sampler (L1 occupancy texture)
//   set0,binding5 storage image    (step count image)
class RayPipeline {
public:
  explicit RayPipeline(const RayPipelineCreateInfo& ci);
  ~RayPipeline();

  RayPipeline(const RayPipeline&) = delete;
  RayPipeline& operator=(const RayPipeline&) = delete;

  VkPipeline             pipeline()    const { return pipeline_; }
  VkPipelineLayout       layout()      const { return layout_; }
  VkDescriptorSetLayout  dset_layout() const { return dset_layout_; }

  VkShaderModule load_module(const std::string& path);

private:
  VkDevice              dev_ = VK_NULL_HANDLE;
  VkPipelineLayout      layout_ = VK_NULL_HANDLE;
  VkPipeline            pipeline_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout dset_layout_ = VK_NULL_HANDLE;
};

} // namespace engine
