#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <vector>

namespace engine {

class RayPipeline;
class PresentPipeline;

struct BrickParams {
  glm::ivec3 brickCoord{0};
  int brickIndex = 0;
};

struct DrawCtx {
  RayPipeline *ray_pipe = nullptr;
  VkDescriptorSet ray_dset = VK_NULL_HANDLE;
  PresentPipeline *light_pipe = nullptr;
  VkDescriptorSet light_dset = VK_NULL_HANDLE;
  VkPipeline comp_pipe = VK_NULL_HANDLE;
  VkPipelineLayout comp_layout = VK_NULL_HANDLE;
  VkDescriptorSet comp_set = VK_NULL_HANDLE;
  VkPipeline comp_l1_pipe = VK_NULL_HANDLE;
  VkPipelineLayout comp_l1_layout = VK_NULL_HANDLE;
  std::vector<VkDescriptorSet> comp_l1_sets;
  std::vector<BrickParams> brick_params; // per-brick push constants
  VkImage occ_image = VK_NULL_HANDLE;
  VkImage mat_image = VK_NULL_HANDLE;
  VkImage occ_l1_image = VK_NULL_HANDLE;
  VkImage occ_l2_image = VK_NULL_HANDLE;
  VkImage brick_ptr_image = VK_NULL_HANDLE;
  VkImage g_albedo = VK_NULL_HANDLE;
  VkImage g_normal = VK_NULL_HANDLE;
  VkImage g_depth = VK_NULL_HANDLE;
  VkImageView g_albedo_view = VK_NULL_HANDLE;
  VkImageView g_normal_view = VK_NULL_HANDLE;
  VkImageView g_depth_view = VK_NULL_HANDLE;
  VkImage steps_image = VK_NULL_HANDLE;
  VkImageView steps_view = VK_NULL_HANDLE;
  VkBuffer steps_buffer = VK_NULL_HANDLE;
  VkImageLayout occ_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  VkImageLayout mat_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  VkImageLayout occ_l1_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  VkImageLayout occ_l2_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  VkImageLayout g_albedo_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  VkImageLayout g_normal_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  VkImageLayout g_depth_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  VkImageLayout steps_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  VkExtent2D steps_dim{0, 0};
  VkExtent2D ray_extent{0, 0};
  VkExtent3D occ_dim{0, 0, 0};
  VkExtent3D dispatch_dim{0, 0, 0};
  VkExtent3D occ_l1_dim{0, 0, 0};
  VkExtent3D dispatch_l1_dim{0, 0, 0};
  VkExtent3D occ_l2_dim{0, 0, 0};
  VkExtent3D dispatch_l2_dim{0, 0, 0};
  VkImageView occ_view = VK_NULL_HANDLE;
  VkImageView occ_l1_view = VK_NULL_HANDLE;
  VkImageView occ_l1_storage_view = VK_NULL_HANDLE;
  VkImageView occ_l2_view = VK_NULL_HANDLE;
  VkImageView occ_l2_storage_view = VK_NULL_HANDLE;
  uint32_t occ_levels = 0;
  bool first_frame = true;
};

struct CameraUBO {
  glm::mat4 inv_view_proj{1.0f};
  glm::vec2 render_resolution{0.0f};
  glm::vec2 output_resolution{0.0f};
  float time = 0.0f;
  float debug_normals = 0.0f;
  float debug_level = 0.0f;
  float debug_steps = 0.0f;
  glm::vec4 pad{0.0f};
};

struct VoxelAABB {
  glm::vec3 min{0.0f};
  float pad0 = 0.0f;
  glm::vec3 max{0.0f};
  float pad1 = 0.0f;
  glm::ivec3 dim{0};
  int pad2 = 0;
  glm::ivec3 occL1Dim{0};
  int pad3 = 0;
  glm::vec3 occL1CellSize{0.0f};
  float pad4 = 0.0f;
  glm::ivec3 occL2Dim{0};
  int pad5 = 0;
  glm::vec3 occL2CellSize{0.0f};
  float pad6 = 0.0f;
};

struct VoxParams {
  glm::ivec3 dim{0};
  int frame = 0;
  glm::vec3 volMin{0.0f};
  float pad0 = 0.0f;
  glm::vec3 volMax{0.0f};
  float pad1 = 0.0f;
  glm::vec3 boxCenter{0.0f};
  float pad2 = 0.0f;
  glm::vec3 boxHalf{0.0f};
  float pad3 = 0.0f;
  glm::vec3 sphereCenter{0.0f};
  float sphereRadius = 0.0f;
  int mode = 0;
  int op = 0;
  int noiseSeed = 0;
  int material = 0;
  float terrainFreq = 0.0f;
  float grassDensity = 0.0f;
  float treeDensity = 0.0f;
  float flowerDensity = 0.0f;
};

struct BuildOccParams {
  glm::ivec3 srcDim{0};
  glm::ivec3 dstDim{0};
  int blockSize = 1;
};

void record_present(VkCommandBuffer cmd, VkImage swap_img, VkImageView view,
                    VkFormat format, VkExtent2D extent, void *user);

} // namespace engine

