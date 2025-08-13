// [UNCHANGED includes...]
#include <engine/camera.hpp>
#include <engine/config.hpp>
#include <engine/engine.hpp>
#include <engine/gfx/memory.hpp>
#include <engine/gfx/present_pipeline.hpp>
#include <engine/gfx/ray_pipeline.hpp>
#include <engine/gfx/vulkan_commands.hpp>
#include <engine/gfx/vulkan_device.hpp>
#include <engine/gfx/vulkan_instance.hpp>
#include <engine/gfx/vulkan_surface.hpp>
#include <engine/gfx/vulkan_swapchain.hpp>
#include <engine/gfx/unique_handles.hpp>
#include <engine/log.hpp>
#include <engine/platform/window.hpp>
#include <engine/vk_checks.hpp>

#include <GLFW/glfw3.h>
#include <algorithm>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <glm/gtc/matrix_inverse.hpp>
#include <memory>
#include <spdlog/spdlog.h>
#include <thread>
#include <vector>
#include <unordered_map>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace engine {

static std::filesystem::path exe_dir() {
#ifdef _WIN32
  char buf[MAX_PATH]{};
  DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
  return std::filesystem::path(std::string(buf, buf + n)).parent_path();
#else
  return std::filesystem::current_path();
#endif
}

static constexpr uint32_t kStepsDown = 4;
static constexpr float kRenderScale = 0.66f;

static constexpr float kColorCompute[4]  = {0.0f, 0.6f, 1.0f, 1.0f};
static constexpr float kColorGraphics[4] = {1.0f, 0.4f, 0.0f, 1.0f};
static constexpr float kColorTransfer[4] = {0.4f, 1.0f, 0.4f, 1.0f};

static float halton(uint32_t i, uint32_t b) {
  float f = 1.0f;
  float r = 0.0f;
  while (i > 0) {
    f /= static_cast<float>(b);
    r += f * static_cast<float>(i % b);
    i /= b;
  }
  return r;
}

// Manager for streaming 3-D voxel bricks.  Each brick is a fixed size volume
// stored inside a larger atlas texture.  Bricks are addressed by integer
// coordinates in world space.  A separate lookup texture stores the mapping
// from world-space brick coordinates to atlas indices.
struct BrickManager {
  struct BrickKey {
    int x = 0, y = 0, z = 0;
    bool operator==(const BrickKey &o) const {
      return x == o.x && y == o.y && z == o.z;
    }
  };
  struct BrickKeyHash {
    size_t operator()(const BrickKey &k) const noexcept {
      size_t h = std::hash<int>()(k.x);
      h ^= std::hash<int>()(k.y + 0x9e3779b9 + (h << 6) + (h >> 2));
      h ^= std::hash<int>()(k.z + 0x9e3779b9 + (h << 6) + (h >> 2));
      return h;
    }
  };

  struct Brick {
    BrickKey key{};
    uint32_t index = 0;
    uint32_t last_used = 0;
  };

  VmaAllocator allocator = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  uint32_t brick_size = 32;      // Edge length of a brick in voxels
  uint32_t max_bricks = 64;      // Maximum number of bricks resident
  glm::ivec3 map_dim{64};        // Dimensions of lookup texture in bricks

  Image3D occ_atlas{};           // Atlas containing occupancy bricks
  Image3D mat_atlas{};           // Atlas containing material bricks
  Image3D index_img{};           // Lookup texture mapping coords->indices

  UniqueImageView occ_view;            // Sampling view for occupancy atlas
  UniqueImageView occ_storage_view;    // Storage view for occupancy atlas
  UniqueImageView mat_view;            // Sampling view for material atlas
  UniqueImageView mat_storage_view;    // Storage view for material atlas
  UniqueImageView index_view;          // Sampling view for index texture
  UniqueImageView index_storage_view;  // Storage view for index texture

  std::unordered_map<BrickKey, Brick, BrickKeyHash> bricks;
  std::vector<Brick> lru;              // Indexed by brick index
  std::vector<uint32_t> free_list;     // Available atlas slots
  std::vector<uint32_t> cpu_index;     // CPU copy of lookup texture

  void init(VmaAllocator alloc, VkDevice dev, uint32_t bsize,
            uint32_t maxb, glm::ivec3 map_dim_ = glm::ivec3(64)) {
    allocator = alloc;
    device = dev;
    brick_size = bsize;
    max_bricks = maxb;
    map_dim = map_dim_;

    occ_atlas = create_image3d(
        allocator, brick_size, brick_size, brick_size * max_bricks,
        VK_FORMAT_R8_UINT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    mat_atlas = create_image3d(
        allocator, brick_size, brick_size, brick_size * max_bricks,
        VK_FORMAT_R8_UINT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    index_img = create_image3d(
        allocator, map_dim.x, map_dim.y, map_dim.z, VK_FORMAT_R32_UINT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.viewType = VK_IMAGE_VIEW_TYPE_3D;
    vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vi.subresourceRange.levelCount = 1;
    vi.subresourceRange.layerCount = 1;

    vi.image = occ_atlas.image;
    vi.format = VK_FORMAT_R8_UINT;
    VK_CHECK(vkCreateImageView(device, &vi, nullptr,
                               occ_view.init(device)));
    VK_CHECK(vkCreateImageView(device, &vi, nullptr,
                               occ_storage_view.init(device)));

    vi.image = mat_atlas.image;
    vi.format = VK_FORMAT_R8_UINT;
    VK_CHECK(vkCreateImageView(device, &vi, nullptr,
                               mat_view.init(device)));
    VK_CHECK(vkCreateImageView(device, &vi, nullptr,
                               mat_storage_view.init(device)));

    vi.image = index_img.image;
    vi.format = VK_FORMAT_R32_UINT;
    VK_CHECK(vkCreateImageView(device, &vi, nullptr,
                               index_view.init(device)));
    VK_CHECK(vkCreateImageView(device, &vi, nullptr,
                               index_storage_view.init(device)));

    cpu_index.resize(static_cast<size_t>(map_dim.x) * map_dim.y * map_dim.z,
                     0xFFFFFFFFu);
    free_list.reserve(max_bricks);
    lru.resize(max_bricks);
    for (uint32_t i = 0; i < max_bricks; ++i)
      free_list.push_back(max_bricks - 1 - i);
  }

  void destroy() {
    index_storage_view.reset();
    index_view.reset();
    mat_storage_view.reset();
    mat_view.reset();
    occ_storage_view.reset();
    occ_view.reset();
    destroy_image3d(allocator, index_img);
    destroy_image3d(allocator, mat_atlas);
    destroy_image3d(allocator, occ_atlas);
  }

  // Acquire a brick at the given coordinate.  If necessary the least recently
  // used brick is recycled.
  uint32_t acquire(const BrickKey &key, uint32_t frame) {
    auto it = bricks.find(key);
    if (it != bricks.end()) {
      lru[it->second.index].last_used = frame;
      return it->second.index;
    }

    uint32_t slot = 0;
    if (!free_list.empty()) {
      slot = free_list.back();
      free_list.pop_back();
    } else {
      // recycle least recently used brick
      uint32_t lru_idx = 0;
      uint32_t lru_frame = lru[0].last_used;
      for (uint32_t i = 1; i < max_bricks; ++i) {
        if (lru[i].last_used < lru_frame) {
          lru_frame = lru[i].last_used;
          lru_idx = i;
        }
      }
      slot = lru_idx;
      bricks.erase(lru[lru_idx].key);
    }
    Brick b{};
    b.key = key;
    b.index = slot;
    b.last_used = frame;
    lru[slot] = b;
    bricks[key] = b;

    // Update CPU lookup table
    size_t offset =
        (static_cast<size_t>(key.z) * map_dim.y + key.y) * map_dim.x + key.x;
    if (offset < cpu_index.size()) cpu_index[offset] = slot;
    return slot;
  }

  // Remove bricks that are further than 'radius' bricks from the camera.
  void stream(const glm::vec3 &cam_pos, uint32_t frame, int radius) {
    BrickKey cam_key{static_cast<int>(std::floor(cam_pos.x / brick_size)),
                     static_cast<int>(std::floor(cam_pos.y / brick_size)),
                     static_cast<int>(std::floor(cam_pos.z / brick_size))};

    for (auto it = bricks.begin(); it != bricks.end();) {
      const BrickKey &bk = it->first;
      int dx = std::abs(bk.x - cam_key.x);
      int dy = std::abs(bk.y - cam_key.y);
      int dz = std::abs(bk.z - cam_key.z);
      if (dx > radius || dy > radius || dz > radius) {
        free_list.push_back(it->second.index);
        size_t offset =
            (static_cast<size_t>(bk.z) * map_dim.y + bk.y) * map_dim.x + bk.x;
        if (offset < cpu_index.size()) cpu_index[offset] = 0xFFFFFFFFu;
        it = bricks.erase(it);
      } else {
        lru[it->second.index].last_used = frame;
        ++it;
      }
    }
  }

  VkImage occ_image() const { return occ_atlas.image; }
  VkImage mat_image() const { return mat_atlas.image; }
  VkImage index_image() const { return index_img.image; }
};

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

static VkShaderModule load_module(VkDevice dev, const std::string &path) {
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

static void record_present(VkCommandBuffer cmd, VkImage, VkImageView view,
                           VkFormat, VkExtent2D extent, void *user) {
  auto *ctx = static_cast<DrawCtx *>(user);
  auto begin_label = [&](const char* name, const float color[4]) {
    if (!cfg::kGpuMarkers || !vkCmdBeginDebugUtilsLabelEXT) return;
    VkDebugUtilsLabelEXT l{VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
    l.pLabelName = name;
    for (int i = 0; i < 4; ++i) l.color[i] = color[i];
    vkCmdBeginDebugUtilsLabelEXT(cmd, &l);
  };
  auto end_label = [&]() {
    if (cfg::kGpuMarkers && vkCmdEndDebugUtilsLabelEXT)
      vkCmdEndDebugUtilsLabelEXT(cmd);
  };

  begin_label("Voxel generation", kColorCompute);
  VkImageMemoryBarrier2 pre[3]{};
  for (int i = 0; i < 3; i++) {
    pre[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    pre[i].srcStageMask =
        ctx->first_frame ? VK_PIPELINE_STAGE_2_NONE : VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    pre[i].srcAccessMask =
        ctx->first_frame ? VK_ACCESS_2_NONE : VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    pre[i].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    pre[i].dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    pre[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    pre[i].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    pre[i].subresourceRange.levelCount = 1;
    pre[i].subresourceRange.layerCount = 1;
  }
  pre[0].image = ctx->occ_image;
  pre[0].oldLayout = ctx->occ_layout;
  pre[1].image = ctx->mat_image;
  pre[1].oldLayout = ctx->mat_layout;
  pre[2].image = ctx->occ_l1_image;
  pre[2].oldLayout = ctx->occ_l1_layout;

  VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dep.imageMemoryBarrierCount = 3;
  dep.pImageMemoryBarriers = pre;
  vkCmdPipelineBarrier2(cmd, &dep);

  ctx->occ_layout = VK_IMAGE_LAYOUT_GENERAL;
  ctx->mat_layout = VK_IMAGE_LAYOUT_GENERAL;
  ctx->occ_l1_layout = VK_IMAGE_LAYOUT_GENERAL;

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->comp_pipe);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->comp_layout,
                          0, 1, &ctx->comp_set, 0, nullptr);
  uint32_t gx = (ctx->dispatch_dim.width + 7) / 8;
  uint32_t gy = (ctx->dispatch_dim.height + 7) / 8;
  uint32_t gz = (ctx->dispatch_dim.depth + 7) / 8;
  for (const auto &bp : ctx->brick_params) {
    vkCmdPushConstants(cmd, ctx->comp_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(BrickParams), &bp);
    vkCmdDispatch(cmd, gx, gy, gz);
  }
  end_label();

  begin_label("Build occupancy", kColorCompute);
  // Transition L0 outputs for sampling
  VkImageMemoryBarrier2 mid[2]{};
  for (int i = 0; i < 2; i++) {
    mid[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    mid[i].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mid[i].srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    mid[i].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mid[i].dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    mid[i].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    mid[i].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    mid[i].subresourceRange.levelCount = 1;
    mid[i].subresourceRange.layerCount = 1;
  }
  mid[0].image = ctx->occ_image;
  mid[0].oldLayout = ctx->occ_layout;
  mid[1].image = ctx->mat_image;
  mid[1].oldLayout = ctx->mat_layout;
  dep.imageMemoryBarrierCount = 2;
  dep.pImageMemoryBarriers = mid;
  vkCmdPipelineBarrier2(cmd, &dep);

  ctx->occ_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  ctx->mat_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

  VkImage occ_images[3] = {ctx->occ_image, ctx->occ_l1_image, ctx->occ_l2_image};
  VkExtent3D occ_dims[3] = {ctx->occ_dim, ctx->occ_l1_dim, ctx->occ_l2_dim};
  VkImageLayout* occ_layouts[3] = {&ctx->occ_layout, &ctx->occ_l1_layout, &ctx->occ_l2_layout};

  for (uint32_t level = 0; level < ctx->occ_levels - 1; ++level) {
    VkImageMemoryBarrier2 b[2]{};
    for (int i = 0; i < 2; ++i) {
      b[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
      b[i].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      b[i].subresourceRange.levelCount = 1;
      b[i].subresourceRange.layerCount = 1;
    }
    b[0].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    b[0].srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    b[0].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    b[0].dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    b[0].oldLayout = *occ_layouts[level];
    b[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b[0].image = occ_images[level];

    b[1].srcStageMask = ctx->first_frame ? VK_PIPELINE_STAGE_2_NONE : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    b[1].srcAccessMask = ctx->first_frame ? VK_ACCESS_2_NONE : VK_ACCESS_2_SHADER_WRITE_BIT;
    b[1].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    b[1].dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    b[1].oldLayout = *occ_layouts[level + 1];
    b[1].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b[1].image = occ_images[level + 1];

    dep.imageMemoryBarrierCount = 2;
    dep.pImageMemoryBarriers = b;
    vkCmdPipelineBarrier2(cmd, &dep);

    *occ_layouts[level] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    *occ_layouts[level + 1] = VK_IMAGE_LAYOUT_GENERAL;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->comp_l1_pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->comp_l1_layout, 0, 1,
                            &ctx->comp_l1_sets[level], 0, nullptr);

    BuildOccParams params{};
    params.srcDim = {static_cast<int>(occ_dims[level].width),
                     static_cast<int>(occ_dims[level].height),
                     static_cast<int>(occ_dims[level].depth)};
    params.dstDim = {static_cast<int>(occ_dims[level + 1].width),
                     static_cast<int>(occ_dims[level + 1].height),
                     static_cast<int>(occ_dims[level + 1].depth)};
    params.blockSize = static_cast<int>(occ_dims[level].width /
                                        occ_dims[level + 1].width);
    vkCmdPushConstants(cmd, ctx->comp_l1_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(params), &params);

    gx = (occ_dims[level + 1].width + 7) / 8;
    gy = (occ_dims[level + 1].height + 7) / 8;
    gz = (occ_dims[level + 1].depth + 7) / 8;
    vkCmdDispatch(cmd, gx, gy, gz);
  }

  end_label();

  begin_label("Geometry", kColorGraphics);

  VkImageMemoryBarrier2 geom[6]{};

  // occ_l1_image: compute write -> fragment sampled read
  geom[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
  geom[0].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
  geom[0].srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
  geom[0].dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
  geom[0].dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
  geom[0].oldLayout = ctx->occ_l1_layout;
  geom[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  geom[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  geom[0].subresourceRange.levelCount = 1;
  geom[0].subresourceRange.layerCount = 1;
  geom[0].image = ctx->occ_l1_image;

  // occ_l2_image: compute write -> fragment sampled read
  geom[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
  geom[1].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
  geom[1].srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
  geom[1].dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
  geom[1].dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
  geom[1].oldLayout = ctx->occ_l2_layout;
  geom[1].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  geom[1].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  geom[1].subresourceRange.levelCount = 1;
  geom[1].subresourceRange.layerCount = 1;
  geom[1].image = ctx->occ_l2_image;

  // steps image for fragment writes
  geom[2].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
  geom[2].srcStageMask = ctx->first_frame ? VK_PIPELINE_STAGE_2_NONE : VK_PIPELINE_STAGE_2_TRANSFER_BIT;
  geom[2].srcAccessMask = ctx->first_frame ? VK_ACCESS_2_NONE : VK_ACCESS_2_TRANSFER_WRITE_BIT;
  geom[2].dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
  geom[2].dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
  geom[2].oldLayout = ctx->steps_layout;
  geom[2].newLayout = VK_IMAGE_LAYOUT_GENERAL;
  geom[2].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  geom[2].subresourceRange.levelCount = 1;
  geom[2].subresourceRange.layerCount = 1;
  geom[2].image = ctx->steps_image;

  // G-buffer images for color attachment writes
  VkImageLayout* g_layouts[3] = {&ctx->g_albedo_layout, &ctx->g_normal_layout,
                                 &ctx->g_depth_layout};
  VkImage g_images[3] = {ctx->g_albedo, ctx->g_normal, ctx->g_depth};
  for (int i = 0; i < 3; i++) {
    int idx = i + 3;
    geom[idx].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    geom[idx].srcStageMask = ctx->first_frame ? VK_PIPELINE_STAGE_2_NONE : VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    geom[idx].srcAccessMask = ctx->first_frame ? VK_ACCESS_2_NONE : VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    geom[idx].dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    geom[idx].dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    geom[idx].oldLayout = *g_layouts[i];
    geom[idx].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    geom[idx].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    geom[idx].subresourceRange.levelCount = 1;
    geom[idx].subresourceRange.layerCount = 1;
    geom[idx].image = g_images[i];
  }

  dep.imageMemoryBarrierCount = 6;
  dep.pImageMemoryBarriers = geom;
  vkCmdPipelineBarrier2(cmd, &dep);

  ctx->occ_l1_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  ctx->occ_l2_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  ctx->steps_layout = VK_IMAGE_LAYOUT_GENERAL;
  ctx->g_albedo_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  ctx->g_normal_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  ctx->g_depth_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  // Geometry pass writing G-buffer
  VkRenderingAttachmentInfo gAtt[3]{
      {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO},
      {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO},
      {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO}};
  VkClearValue gclear{};
  gclear.color = {{0, 0, 0, 0}};
  gAtt[0].imageView = ctx->g_albedo_view;
  gAtt[0].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  gAtt[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  gAtt[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  gAtt[0].clearValue = gclear;
  gAtt[1].imageView = ctx->g_normal_view;
  gAtt[1].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  gAtt[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  gAtt[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  gAtt[1].clearValue = gclear;
  gAtt[2].imageView = ctx->g_depth_view;
  gAtt[2].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  gAtt[2].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  gAtt[2].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  gAtt[2].clearValue = gclear;

  VkRenderingInfo gi{VK_STRUCTURE_TYPE_RENDERING_INFO};
  gi.renderArea.offset = {0, 0};
  gi.renderArea.extent = ctx->ray_extent;
  gi.layerCount = 1;
  gi.colorAttachmentCount = 3;
  gi.pColorAttachments = gAtt;

  vkCmdBeginRendering(cmd, &gi);

  // Geometry pass renders the voxel scene at reduced resolution (ray_extent)
  VkViewport vp_lo{};
  vp_lo.x = 0.0f;
  vp_lo.y = static_cast<float>(ctx->ray_extent.height);
  vp_lo.width = static_cast<float>(ctx->ray_extent.width);
  vp_lo.height = -static_cast<float>(ctx->ray_extent.height);
  vp_lo.minDepth = 0.0f;
  vp_lo.maxDepth = 1.0f;
  vkCmdSetViewport(cmd, 0, 1, &vp_lo);
  VkRect2D sc_lo{{0, 0}, ctx->ray_extent};
  vkCmdSetScissor(cmd, 0, 1, &sc_lo);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    ctx->ray_pipe->pipeline());
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          ctx->ray_pipe->layout(), 0, 1, &ctx->ray_dset, 0,
                          nullptr);
  vkCmdDraw(cmd, 3, 1, 0, 0);
  vkCmdEndRendering(cmd);
  end_label();

  begin_label("Readback", kColorTransfer);
  VkImageSubresourceRange stepsRange{};
  stepsRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  stepsRange.levelCount = 1;
  stepsRange.layerCount = 1;

  VkImageMemoryBarrier2 steps_to_copy{};
  steps_to_copy.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
  steps_to_copy.srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
  steps_to_copy.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
  steps_to_copy.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
  steps_to_copy.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
  steps_to_copy.oldLayout = ctx->steps_layout;
  steps_to_copy.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  steps_to_copy.subresourceRange = stepsRange;
  steps_to_copy.image = ctx->steps_image;

  VkImageMemoryBarrier2 gpost[3]{};
  VkImageLayout* g_layouts2[3] = {&ctx->g_albedo_layout, &ctx->g_normal_layout,
                                  &ctx->g_depth_layout};
  VkImage g_images2[3] = {ctx->g_albedo, ctx->g_normal, ctx->g_depth};
  for (int i = 0; i < 3; i++) {
    gpost[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    gpost[i].srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    gpost[i].srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    gpost[i].dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    gpost[i].dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    gpost[i].oldLayout = *g_layouts2[i];
    gpost[i].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    gpost[i].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    gpost[i].subresourceRange.levelCount = 1;
    gpost[i].subresourceRange.layerCount = 1;
    gpost[i].image = g_images2[i];
  }

  VkImageMemoryBarrier2 readback_barriers[4] = {steps_to_copy, gpost[0], gpost[1], gpost[2]};
  dep.imageMemoryBarrierCount = 4;
  dep.pImageMemoryBarriers = readback_barriers;
  vkCmdPipelineBarrier2(cmd, &dep);

  ctx->steps_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  for (int i = 0; i < 3; ++i)
    *g_layouts2[i] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

  VkBufferImageCopy bic{};
  bic.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  bic.imageSubresource.layerCount = 1;
  bic.imageExtent = {ctx->steps_dim.width, ctx->steps_dim.height, 1};
  vkCmdCopyImageToBuffer(cmd, ctx->steps_image,
                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         ctx->steps_buffer, 1, &bic);

  VkImageMemoryBarrier2 steps_to_clear{};
  steps_to_clear.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
  steps_to_clear.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
  steps_to_clear.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
  steps_to_clear.dstStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
  steps_to_clear.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
  steps_to_clear.oldLayout = ctx->steps_layout;
  steps_to_clear.newLayout = VK_IMAGE_LAYOUT_GENERAL;
  steps_to_clear.subresourceRange = stepsRange;
  steps_to_clear.image = ctx->steps_image;
  dep.imageMemoryBarrierCount = 1;
  dep.pImageMemoryBarriers = &steps_to_clear;
  vkCmdPipelineBarrier2(cmd, &dep);

  ctx->steps_layout = VK_IMAGE_LAYOUT_GENERAL;

  VkClearColorValue zero{{0, 0, 0, 0}};
  vkCmdClearColorImage(cmd, ctx->steps_image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1,
                       &stepsRange);
  end_label();

  begin_label("Postprocess", kColorGraphics);

  // Lighting pass to final image
  VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  color.imageView = view;
  color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  VkClearValue clear;
  clear.color = {{0.06f, 0.07f, 0.10f, 1.0f}};
  color.clearValue = clear;

  VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
  ri.renderArea.offset = {0, 0};
  ri.renderArea.extent = extent;
  ri.layerCount = 1;
  ri.colorAttachmentCount = 1;
  ri.pColorAttachments = &color;

  // Final lighting/present pass upscales to the swapchain extent
  vkCmdBeginRendering(cmd, &ri);

  VkViewport vp_hi{};
  vp_hi.x = 0.0f;
  vp_hi.y = static_cast<float>(extent.height);
  vp_hi.width = static_cast<float>(extent.width);
  vp_hi.height = -static_cast<float>(extent.height);
  vp_hi.minDepth = 0.0f;
  vp_hi.maxDepth = 1.0f;
  vkCmdSetViewport(cmd, 0, 1, &vp_hi);

  VkRect2D sc_hi{{0, 0}, extent};
  vkCmdSetScissor(cmd, 0, 1, &sc_hi);

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    ctx->light_pipe->pipeline());
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          ctx->light_pipe->layout(), 0, 1, &ctx->light_dset, 0,
                          nullptr);
  vkCmdDraw(cmd, 3, 1, 0, 0);
  vkCmdEndRendering(cmd);
  end_label();

  ctx->first_frame = false;
}

int run() {
  init_logging();
  log_boot_banner("engine");

  WindowDesc wdesc{};
  wdesc.title = "vk_engine fullscreen present";
  auto window = create_window(wdesc);

  Camera cam;
  GLFWwindow *glfw_win = static_cast<GLFWwindow *>(window->native_handle());
  glfwSetInputMode(glfw_win, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
  double last_x = 0.0, last_y = 0.0;
  glfwGetCursorPos(glfw_win, &last_x, &last_y);
  auto last_time = std::chrono::high_resolution_clock::now();

  VulkanInstanceCreateInfo ici{};
  ici.enable_validation = cfg::kValidation;
  for (auto *e : platform_required_instance_extensions())
    ici.extra_extensions.push_back(e);
  VulkanInstance instance{ici};
  VulkanSurface surface{instance.vk(), window->native_handle()};
  VulkanDeviceCreateInfo dci{};
  dci.instance = instance.vk();
  dci.surface = surface.vk();
  dci.enable_validation = cfg::kValidation;
  VulkanDevice device{dci};
  load_debug_label_functions(instance.vk(), device.device());

  const auto shader_dir = exe_dir() / "shaders";
  const auto vs_path = (shader_dir / "vs_fullscreen.vert.spv").string();
  const auto ray_fs_path = (shader_dir / "fs_raycast.frag.spv").string();
  const auto light_fs_path = (shader_dir / "fs_lighting.frag.spv").string();
  spdlog::info("[vk] Using shaders: {}", shader_dir.string());

  GpuAllocator allocator;
  allocator.init(instance.vk(), device.physical(), device.device());
  {
    TransferContext transfer{device.device(), device.graphics_family(),
                             allocator.raw()};

  // Samplers for sampled images
  UniqueSampler linear_sampler;
  UniqueSampler nearest_sampler;
  {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(device.physical(), &props);
    VkPhysicalDeviceFeatures feats{};
    vkGetPhysicalDeviceFeatures(device.physical(), &feats);

    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.addressModeU = si.addressModeV = si.addressModeW =
        VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.minLod = 0.0f;
    si.maxLod = 0.0f;

    // linear sampler for floating point images
    si.magFilter = VK_FILTER_LINEAR;
    si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    if (feats.samplerAnisotropy) {
      si.anisotropyEnable = VK_TRUE;
      si.maxAnisotropy = props.limits.maxSamplerAnisotropy;
    } else {
      si.anisotropyEnable = VK_FALSE;
      si.maxAnisotropy = 1.0f;
    }
    VK_CHECK(
        vkCreateSampler(device.device(), &si, nullptr, linear_sampler.init(device.device())));

    // nearest sampler for integer textures (e.g. occupancy grid)
    si.magFilter = VK_FILTER_NEAREST;
    si.minFilter = VK_FILTER_NEAREST;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.anisotropyEnable = VK_FALSE;
    si.maxAnisotropy = 1.0f;
    VK_CHECK(
        vkCreateSampler(device.device(), &si, nullptr, nearest_sampler.init(device.device())));
  }

  // Buffers and occupancy texture
  Buffer cam_buf = create_buffer(allocator.raw(), sizeof(CameraUBO),
                                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
  Buffer vox_buf = create_buffer(allocator.raw(), sizeof(VoxelAABB),
                                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
  Buffer vox_params_buf = create_buffer(allocator.raw(), sizeof(VoxParams),
                                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
  const uint32_t N = 128;
  const uint32_t N1 = N / 4;
  const uint32_t N2 = N / 32;
  BrickManager brick_mgr;
  Image3D occ_l1_img{};
  Image3D occ_l2_img{};
  UniqueImageView occ_l1_view;
  UniqueImageView occ_l1_storage_view;
  UniqueImageView occ_l2_view;
  UniqueImageView occ_l2_storage_view;
  // brick_mgr provides views for occupancy/material/brick index textures
  {
    brick_mgr.init(allocator.raw(), device.device(), 32, 64, {N2, N2, N2});

    VkImageViewCreateInfo ovi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    ovi.viewType = VK_IMAGE_VIEW_TYPE_3D;
    ovi.format = VK_FORMAT_R8_UINT;
    ovi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    ovi.subresourceRange.levelCount = 1;
    ovi.subresourceRange.layerCount = 1;

    occ_l1_img =
        create_image3d(allocator.raw(), N1, N1, N1, VK_FORMAT_R8_UINT,
                       VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    ovi.image = occ_l1_img.image;
    VK_CHECK(vkCreateImageView(device.device(), &ovi, nullptr,
                               occ_l1_view.init(device.device())));
    VK_CHECK(vkCreateImageView(device.device(), &ovi, nullptr,
                               occ_l1_storage_view.init(device.device())));

    occ_l2_img =
        create_image3d(allocator.raw(), N2, N2, N2, VK_FORMAT_R8_UINT,
                       VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    ovi.image = occ_l2_img.image;
    ovi.format = VK_FORMAT_R8_UINT;
    VK_CHECK(vkCreateImageView(device.device(), &ovi, nullptr,
                               occ_l2_view.init(device.device())));
    VK_CHECK(vkCreateImageView(device.device(), &ovi, nullptr,
                               occ_l2_storage_view.init(device.device())));
  }

  // Upload static voxel bounds
  VoxelAABB vubo{};
  vubo.min = {0.0f, 0.0f, 0.0f};
  vubo.max = {static_cast<float>(N), static_cast<float>(N),
              static_cast<float>(N)};
  vubo.dim = {static_cast<int>(N), static_cast<int>(N), static_cast<int>(N)};
  vubo.occL1Dim = {static_cast<int>(N1), static_cast<int>(N1), static_cast<int>(N1)};
  vubo.occL1CellSize =
      (vubo.max - vubo.min) / glm::vec3(vubo.occL1Dim);
  vubo.occL2Dim = {static_cast<int>(N2), static_cast<int>(N2), static_cast<int>(N2)};
  vubo.occL2CellSize =
      (vubo.max - vubo.min) / glm::vec3(vubo.occL2Dim);
  upload_buffer(allocator.raw(), transfer, device.graphics_queue(), vox_buf,
                &vubo, sizeof(vubo));

  // Compute pipeline to generate voxel occupancy and material textures
  UniqueDescriptorSetLayout comp_dsl;
  UniquePipelineLayout comp_layout;
  UniquePipeline comp_pipeline;
  UniqueDescriptorPool comp_pool;
  VkDescriptorSet comp_set = VK_NULL_HANDLE;
  UniqueDescriptorSetLayout comp_l1_dsl;
  UniquePipelineLayout comp_l1_layout;
  UniquePipeline comp_l1_pipeline;
  UniqueDescriptorPool comp_l1_pool;
  DrawCtx ctx{};
  {
    VkDescriptorSetLayoutBinding binds[3]{};
    binds[0].binding = 0;
    binds[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    binds[0].descriptorCount = 1;
    binds[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    binds[1].binding = 1;
    binds[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    binds[1].descriptorCount = 1;
    binds[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    binds[2].binding = 2;
    binds[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    binds[2].descriptorCount = 1;
    binds[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dlci{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dlci.bindingCount = 3;
    dlci.pBindings = binds;
    VK_CHECK(vkCreateDescriptorSetLayout(device.device(), &dlci, nullptr,
                                         comp_dsl.init(device.device())));

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = sizeof(BrickParams);

    VkPipelineLayoutCreateInfo plci{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    VkDescriptorSetLayout comp_dsl_handle = comp_dsl.get();
    plci.pSetLayouts = &comp_dsl_handle;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    VK_CHECK(vkCreatePipelineLayout(device.device(), &plci, nullptr,
                                    comp_layout.init(device.device())));

    const auto cs_path = (shader_dir / "procgen_voxels.comp.spv").string();
    VkShaderModule cs = load_module(device.device(), cs_path);
    VkPipelineShaderStageCreateInfo stage{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = cs;
    stage.pName = "main";
    VkComputePipelineCreateInfo cpci{
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = stage;
    cpci.layout = comp_layout.get();
    VK_CHECK(vkCreateComputePipelines(device.device(), device.pipeline_cache(),
                                      1, &cpci, nullptr,
                                      comp_pipeline.init(device.device())));
    vkDestroyShaderModule(device.device(), cs, nullptr);

    VkDescriptorPoolSize psizes[2]{};
    psizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    psizes[0].descriptorCount = 2;
    psizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    psizes[1].descriptorCount = 1;
    VkDescriptorPoolCreateInfo dpci{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpci.maxSets = 1;
    dpci.poolSizeCount = 2;
    dpci.pPoolSizes = psizes;
    VK_CHECK(vkCreateDescriptorPool(device.device(), &dpci, nullptr,
                                    comp_pool.init(device.device())));

    VkDescriptorSetAllocateInfo ai{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = comp_pool.get();
    ai.descriptorSetCount = 1;
    comp_dsl_handle = comp_dsl.get();
    ai.pSetLayouts = &comp_dsl_handle;
    VK_CHECK(vkAllocateDescriptorSets(device.device(), &ai, &comp_set));

    VkDescriptorImageInfo occ_info{};
    occ_info.imageView = brick_mgr.occ_storage_view.get();
    occ_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorImageInfo mat_info{};
    mat_info.imageView = brick_mgr.mat_storage_view.get();
    mat_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorBufferInfo param_bi{};
    param_bi.buffer = vox_params_buf.buffer;
    param_bi.range = sizeof(VoxParams);
    VkWriteDescriptorSet ws[3]{};
    ws[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    ws[0].dstSet = comp_set;
    ws[0].dstBinding = 0;
    ws[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    ws[0].descriptorCount = 1;
    ws[0].pImageInfo = &occ_info;
    ws[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    ws[1].dstSet = comp_set;
    ws[1].dstBinding = 1;
    ws[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    ws[1].descriptorCount = 1;
    ws[1].pImageInfo = &mat_info;
    ws[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    ws[2].dstSet = comp_set;
    ws[2].dstBinding = 2;
    ws[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    ws[2].descriptorCount = 1;
    ws[2].pBufferInfo = &param_bi;
    vkUpdateDescriptorSets(device.device(), 3, ws, 0, nullptr);
  }

  // Compute pipeline to build coarse L1 occupancy from L0
  {
    VkDescriptorSetLayoutBinding binds[2]{};
    binds[0].binding = 0;
    binds[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    binds[0].descriptorCount = 1;
    binds[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    binds[1].binding = 1;
    binds[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    binds[1].descriptorCount = 1;
    binds[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dlci{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dlci.bindingCount = 2;
    dlci.pBindings = binds;
    VK_CHECK(vkCreateDescriptorSetLayout(device.device(), &dlci, nullptr,
                                         comp_l1_dsl.init(device.device())));

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = sizeof(BuildOccParams);
    VkPipelineLayoutCreateInfo plci{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    VkDescriptorSetLayout comp_l1_dsl_handle = comp_l1_dsl.get();
    plci.pSetLayouts = &comp_l1_dsl_handle;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    VK_CHECK(vkCreatePipelineLayout(device.device(), &plci, nullptr,
                                    comp_l1_layout.init(device.device())));

    const auto cs_path = (shader_dir / "build_occ_l1.comp.spv").string();
    VkShaderModule cs = load_module(device.device(), cs_path);
    VkPipelineShaderStageCreateInfo stage{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = cs;
    stage.pName = "main";
    VkComputePipelineCreateInfo cpci{
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = stage;
    cpci.layout = comp_l1_layout.get();
    VK_CHECK(vkCreateComputePipelines(device.device(), device.pipeline_cache(),
                                      1, &cpci, nullptr,
                                      comp_l1_pipeline.init(device.device())));
    vkDestroyShaderModule(device.device(), cs, nullptr);

    const uint32_t occ_levels = 3;
    VkDescriptorPoolSize psizes[2]{};
    psizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    psizes[0].descriptorCount = occ_levels - 1;
    psizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    psizes[1].descriptorCount = occ_levels - 1;
    VkDescriptorPoolCreateInfo dpci{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpci.maxSets = occ_levels - 1;
    dpci.poolSizeCount = 2;
    dpci.pPoolSizes = psizes;
    VK_CHECK(vkCreateDescriptorPool(device.device(), &dpci, nullptr,
                                    comp_l1_pool.init(device.device())));

    std::vector<VkDescriptorSet> comp_sets(occ_levels - 1);
    std::vector<VkDescriptorSetLayout> layouts(occ_levels - 1, comp_l1_dsl.get());
    VkDescriptorSetAllocateInfo ai{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = comp_l1_pool.get();
    ai.descriptorSetCount = occ_levels - 1;
    ai.pSetLayouts = layouts.data();
    VK_CHECK(vkAllocateDescriptorSets(device.device(), &ai, comp_sets.data()));

    std::vector<VkDescriptorImageInfo> src_infos(occ_levels - 1);
    std::vector<VkDescriptorImageInfo> dst_infos(occ_levels - 1);
    std::vector<VkWriteDescriptorSet> ws(2 * (occ_levels - 1));

    src_infos[0].sampler = nearest_sampler.get();
    src_infos[0].imageView = brick_mgr.occ_view.get();
    src_infos[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    dst_infos[0].imageView = occ_l1_storage_view.get();
    dst_infos[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    src_infos[1].sampler = nearest_sampler.get();
    src_infos[1].imageView = occ_l1_view.get();
    src_infos[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    dst_infos[1].imageView = occ_l2_storage_view.get();
    dst_infos[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    for (uint32_t i = 0; i < occ_levels - 1; ++i) {
      ws[2 * i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      ws[2 * i].dstSet = comp_sets[i];
      ws[2 * i].dstBinding = 0;
      ws[2 * i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      ws[2 * i].descriptorCount = 1;
      ws[2 * i].pImageInfo = &src_infos[i];

      ws[2 * i + 1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      ws[2 * i + 1].dstSet = comp_sets[i];
      ws[2 * i + 1].dstBinding = 1;
      ws[2 * i + 1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      ws[2 * i + 1].descriptorCount = 1;
      ws[2 * i + 1].pImageInfo = &dst_infos[i];
    }
    vkUpdateDescriptorSets(device.device(), static_cast<uint32_t>(ws.size()),
                           ws.data(), 0, nullptr);

    ctx.occ_view = brick_mgr.occ_view.get();
    ctx.occ_l1_view = occ_l1_view.get();
    ctx.occ_l1_storage_view = occ_l1_storage_view.get();
    ctx.occ_l2_view = occ_l2_view.get();
    ctx.occ_l2_storage_view = occ_l2_storage_view.get();
    ctx.occ_levels = occ_levels;
    ctx.comp_l1_sets = comp_sets;
  }

  std::unique_ptr<VulkanSwapchain> swapchain;
  std::unique_ptr<VulkanCommands> commands;
  std::unique_ptr<RayPipeline> ray_pipeline;
  std::unique_ptr<PresentPipeline> present_pipeline;

  UniqueDescriptorPool dpool;
  VkDescriptorSet ray_dset = VK_NULL_HANDLE;
  VkDescriptorSet light_dset = VK_NULL_HANDLE;

  // Descriptor pool for geometry and lighting passes
  {
    VkDescriptorPoolSize sizes[3]{};
    sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    sizes[0].descriptorCount = 3;
    sizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    // Ray pass now samples 5 images (L0 occ, material, L1 occ, L2 occ, brick ptr)
    // and the lighting pass samples 3 G-buffer images, so allocate 8 in total.
    sizes[1].descriptorCount = 8;
    sizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    sizes[2].descriptorCount = 1;
    VkDescriptorPoolCreateInfo dpci{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpci.maxSets = 2;
    dpci.poolSizeCount = 3;
    dpci.pPoolSizes = sizes;
    VK_CHECK(vkCreateDescriptorPool(device.device(), &dpci, nullptr,
                                    dpool.init(device.device())));
  }

  Image2D g_albedo_img{};
  Image2D g_normal_img{};
  Image2D g_depth_img{};
  Image2D steps_img{};
  UniqueImageView g_albedo_view;
  UniqueImageView g_normal_view;
  UniqueImageView g_depth_view;
  UniqueImageView steps_view;
  Buffer steps_buf{};

  auto destroy_swapchain_stack = [&]() {
    if (ray_dset) {
      vkFreeDescriptorSets(device.device(), dpool.get(), 1, &ray_dset);
      ray_dset = VK_NULL_HANDLE;
    }
    if (light_dset) {
      vkFreeDescriptorSets(device.device(), dpool.get(), 1, &light_dset);
      light_dset = VK_NULL_HANDLE;
    }
    g_albedo_view.reset();
    g_normal_view.reset();
    g_depth_view.reset();
    steps_view.reset();
    destroy_image2d(allocator.raw(), g_albedo_img);
    destroy_image2d(allocator.raw(), g_normal_img);
    destroy_image2d(allocator.raw(), g_depth_img);
    destroy_image2d(allocator.raw(), steps_img);
    destroy_buffer(allocator.raw(), steps_buf);
    commands.reset();
    swapchain.reset();
  };

  auto create_swapchain_stack = [&](uint32_t sw, uint32_t sh) {
    VulkanSwapchainCreateInfo sci{};
    sci.physical = device.physical();
    sci.device = device.device();
    sci.surface = surface.vk();
    sci.desired_width = sw;
    sci.desired_height = sh;
    swapchain = std::make_unique<VulkanSwapchain>(sci);

    VulkanCommandsCreateInfo cci{};
    cci.device = device.device();
    cci.graphics_family = device.graphics_family();
    cci.image_count = static_cast<uint32_t>(swapchain->image_views().size());
    commands = std::make_unique<VulkanCommands>(cci);

    if (!ray_pipeline) {
      RayPipelineCreateInfo rpci{};
      rpci.device = device.device();
      rpci.pipeline_cache = device.pipeline_cache();
      rpci.vs_spv = vs_path;
      rpci.fs_spv = ray_fs_path;
      ray_pipeline = std::make_unique<RayPipeline>(rpci);
    }
    if (!present_pipeline ||
        present_pipeline->color_format() != swapchain->image_format()) {
      PresentPipelineCreateInfo pci{};
      pci.device = device.device();
      pci.pipeline_cache = device.pipeline_cache();
      pci.color_format = swapchain->image_format();
      pci.vs_spv = vs_path;
      pci.fs_spv = light_fs_path;
      present_pipeline = std::make_unique<PresentPipeline>(pci);
    }

    if (ray_dset) {
      vkFreeDescriptorSets(device.device(), dpool.get(), 1, &ray_dset);
      ray_dset = VK_NULL_HANDLE;
    }
    if (light_dset) {
      vkFreeDescriptorSets(device.device(), dpool.get(), 1, &light_dset);
      light_dset = VK_NULL_HANDLE;
    }

    VkDescriptorSetAllocateInfo ai{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = dpool.get();
    VkDescriptorSetLayout layout = ray_pipeline->dset_layout();
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &layout;
    VK_CHECK(vkAllocateDescriptorSets(device.device(), &ai, &ray_dset));

    uint32_t rw = static_cast<uint32_t>(static_cast<float>(sw) * kRenderScale);
    uint32_t rh = static_cast<uint32_t>(static_cast<float>(sh) * kRenderScale);
    rw = std::max(1u, rw);
    rh = std::max(1u, rh);
    uint32_t steps_w = (rw + kStepsDown - 1) / kStepsDown;
    uint32_t steps_h = (rh + kStepsDown - 1) / kStepsDown;
    steps_img = create_image2d(
        allocator.raw(), steps_w, steps_h, VK_FORMAT_R32_UINT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
            VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    VkImageViewCreateInfo svi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    svi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    svi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    svi.subresourceRange.levelCount = 1;
    svi.subresourceRange.layerCount = 1;
    svi.image = steps_img.image;
    svi.format = VK_FORMAT_R32_UINT;
    VK_CHECK(vkCreateImageView(device.device(), &svi, nullptr,
                                steps_view.init(device.device())));
    steps_buf = create_host_buffer(allocator.raw(),
                                   steps_w * steps_h * sizeof(uint32_t),
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    VkDescriptorBufferInfo cam_bi{};
    cam_bi.buffer = cam_buf.buffer;
    cam_bi.range = sizeof(CameraUBO);
    VkDescriptorBufferInfo vox_bi{};
    vox_bi.buffer = vox_buf.buffer;
    vox_bi.range = sizeof(VoxelAABB);
    VkDescriptorImageInfo occ_info{};
    occ_info.sampler = nearest_sampler.get();
    occ_info.imageView = brick_mgr.occ_view.get();
    occ_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo mat_info{};
    mat_info.sampler = nearest_sampler.get();
    mat_info.imageView = brick_mgr.mat_view.get();
    mat_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo occ_l1_info{};
    occ_l1_info.sampler = nearest_sampler.get();
    occ_l1_info.imageView = occ_l1_view.get();
    occ_l1_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo occ_l2_info{};
    occ_l2_info.sampler = nearest_sampler.get();
    occ_l2_info.imageView = occ_l2_view.get();
    occ_l2_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo brick_info{};
    brick_info.sampler = nearest_sampler.get();
    brick_info.imageView = brick_mgr.index_view.get();
    brick_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo steps_info{};
    steps_info.imageView = steps_view.get();
    steps_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet rwrites[8]{};
    rwrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    rwrites[0].dstSet = ray_dset;
    rwrites[0].dstBinding = 0;
    rwrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    rwrites[0].descriptorCount = 1;
    rwrites[0].pBufferInfo = &cam_bi;
    rwrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    rwrites[1].dstSet = ray_dset;
    rwrites[1].dstBinding = 1;
    rwrites[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    rwrites[1].descriptorCount = 1;
    rwrites[1].pBufferInfo = &vox_bi;
    rwrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    rwrites[2].dstSet = ray_dset;
    rwrites[2].dstBinding = 2;
    rwrites[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    rwrites[2].descriptorCount = 1;
    rwrites[2].pImageInfo = &occ_info;
    rwrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    rwrites[3].dstSet = ray_dset;
    rwrites[3].dstBinding = 3;
    rwrites[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    rwrites[3].descriptorCount = 1;
    rwrites[3].pImageInfo = &mat_info;
    rwrites[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    rwrites[4].dstSet = ray_dset;
    rwrites[4].dstBinding = 4;
    rwrites[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    rwrites[4].descriptorCount = 1;
    rwrites[4].pImageInfo = &occ_l1_info;
    rwrites[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    rwrites[5].dstSet = ray_dset;
    rwrites[5].dstBinding = 5;
    rwrites[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    rwrites[5].descriptorCount = 1;
    rwrites[5].pImageInfo = &steps_info;
    rwrites[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    rwrites[6].dstSet = ray_dset;
    rwrites[6].dstBinding = 6;
    rwrites[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    rwrites[6].descriptorCount = 1;
    rwrites[6].pImageInfo = &occ_l2_info;
    rwrites[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    rwrites[7].dstSet = ray_dset;
    rwrites[7].dstBinding = 7;
    rwrites[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    rwrites[7].descriptorCount = 1;
    rwrites[7].pImageInfo = &brick_info;
    vkUpdateDescriptorSets(device.device(), 8, rwrites, 0, nullptr);

    g_albedo_img = create_image2d(
        allocator.raw(), rw, rh, VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    g_normal_img = create_image2d(
        allocator.raw(), rw, rh, VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    g_depth_img = create_image2d(allocator.raw(), rw, rh, VK_FORMAT_R32_SFLOAT,
                                 VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                     VK_IMAGE_USAGE_SAMPLED_BIT);

    VkImageViewCreateInfo iv{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    iv.viewType = VK_IMAGE_VIEW_TYPE_2D;
    iv.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    iv.subresourceRange.levelCount = 1;
    iv.subresourceRange.layerCount = 1;
    iv.image = g_albedo_img.image;
    iv.format = VK_FORMAT_R8G8B8A8_UNORM;
    VK_CHECK(vkCreateImageView(device.device(), &iv, nullptr,
                               g_albedo_view.init(device.device())));
    iv.image = g_normal_img.image;
    iv.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    VK_CHECK(vkCreateImageView(device.device(), &iv, nullptr,
                               g_normal_view.init(device.device())));
    iv.image = g_depth_img.image;
    iv.format = VK_FORMAT_R32_SFLOAT;
    VK_CHECK(vkCreateImageView(device.device(), &iv, nullptr,
                               g_depth_view.init(device.device())));

    VkDescriptorSetLayout layout2 = present_pipeline->dset_layout();
    ai.pSetLayouts = &layout2;
    VK_CHECK(vkAllocateDescriptorSets(device.device(), &ai, &light_dset));

    VkDescriptorImageInfo albedo_info{};
    albedo_info.sampler = linear_sampler.get();
    albedo_info.imageView = g_albedo_view.get();
    albedo_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo normal_info{};
    normal_info.sampler = linear_sampler.get();
    normal_info.imageView = g_normal_view.get();
    normal_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo depth_info{};
    depth_info.sampler = linear_sampler.get();
    depth_info.imageView = g_depth_view.get();
    depth_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet lwrites[4]{};
    lwrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    lwrites[0].dstSet = light_dset;
    lwrites[0].dstBinding = 0;
    lwrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    lwrites[0].descriptorCount = 1;
    lwrites[0].pBufferInfo = &cam_bi;
    lwrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    lwrites[1].dstSet = light_dset;
    lwrites[1].dstBinding = 1;
    lwrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    lwrites[1].descriptorCount = 1;
    lwrites[1].pImageInfo = &albedo_info;
    lwrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    lwrites[2].dstSet = light_dset;
    lwrites[2].dstBinding = 2;
    lwrites[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    lwrites[2].descriptorCount = 1;
    lwrites[2].pImageInfo = &normal_info;
    lwrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    lwrites[3].dstSet = light_dset;
    lwrites[3].dstBinding = 3;
    lwrites[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    lwrites[3].descriptorCount = 1;
    lwrites[3].pImageInfo = &depth_info;
    vkUpdateDescriptorSets(device.device(), 4, lwrites, 0, nullptr);

    ctx.ray_pipe = ray_pipeline.get();
    ctx.ray_dset = ray_dset;
    ctx.light_pipe = present_pipeline.get();
    ctx.light_dset = light_dset;
    ctx.g_albedo = g_albedo_img.image;
    ctx.g_normal = g_normal_img.image;
    ctx.g_depth = g_depth_img.image;
    ctx.g_albedo_view = g_albedo_view.get();
    ctx.g_normal_view = g_normal_view.get();
    ctx.g_depth_view = g_depth_view.get();
    ctx.steps_image = steps_img.image;
    ctx.steps_view = steps_view.get();
    ctx.steps_buffer = steps_buf.buffer;
    ctx.steps_dim = {steps_w, steps_h};
    ctx.ray_extent = {rw, rh};
    ctx.g_albedo_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    ctx.g_normal_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    ctx.g_depth_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    ctx.steps_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    ctx.first_frame = true;
  };

  {
    auto fb = window->framebuffer_size();
    create_swapchain_stack(static_cast<uint32_t>(fb.first),
                           static_cast<uint32_t>(fb.second));
  }

  ctx.comp_pipe = comp_pipeline.get();
  ctx.comp_layout = comp_layout.get();
  ctx.comp_set = comp_set;
  ctx.comp_l1_pipe = comp_l1_pipeline.get();
  ctx.comp_l1_layout = comp_l1_layout.get();
  ctx.occ_image = brick_mgr.occ_image();
  ctx.mat_image = brick_mgr.mat_image();
  ctx.occ_l1_image = occ_l1_img.image;
  ctx.occ_l2_image = occ_l2_img.image;
  ctx.occ_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  ctx.mat_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  ctx.occ_l1_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  ctx.occ_l2_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  ctx.brick_ptr_image = brick_mgr.index_image();
  ctx.occ_dim = {brick_mgr.brick_size, brick_mgr.brick_size,
                 brick_mgr.brick_size * brick_mgr.max_bricks};
  ctx.dispatch_dim = {brick_mgr.brick_size, brick_mgr.brick_size,
                      brick_mgr.brick_size};
  ctx.occ_l1_dim = {ctx.occ_dim.width / 4, ctx.occ_dim.height / 4,
                    ctx.occ_dim.depth / 4};
  ctx.dispatch_l1_dim = ctx.occ_l1_dim;
  ctx.occ_l2_dim = {ctx.occ_dim.width / 32, ctx.occ_dim.height / 32,
                    ctx.occ_dim.depth / 32};
  ctx.dispatch_l2_dim = ctx.occ_l2_dim;
  ctx.first_frame = true;
  VkExtent2D last = swapchain->extent();
  float total_time = 0.0f;
  int frame_counter = 0;
  uint32_t jitter_index = 0;
  using namespace std::chrono_literals;

  enum {
    MODE_CLEAR = 0,
    MODE_FILL_BOX = 1,
    MODE_FILL_SPHERE = 2,
    MODE_NOISE = 3,
    MODE_TERRAIN = 4
  };
  enum { OP_REPLACE = 0, OP_UNION = 1, OP_INTERSECTION = 2, OP_SUBTRACT = 3 };

  glm::vec3 sphere_center{static_cast<float>(N) / 2.0f,
                          static_cast<float>(N) / 2.0f,
                          static_cast<float>(N) / 2.0f};
  float sphere_radius = 30.0f;
  glm::vec3 box_center{60.0f, 60.0f, 60.0f};
  glm::vec3 box_half{40.0f, 40.0f, 40.0f};
  int vox_mode = MODE_TERRAIN;
  int vox_op = OP_REPLACE;
  int noise_seed = 0;

  while (!window->should_close()) {
    window->poll_events();

    auto now = std::chrono::high_resolution_clock::now();
    float dt = std::chrono::duration<float>(now - last_time).count();
    last_time = now;
    total_time += dt;

    double cx, cy;
    glfwGetCursorPos(glfw_win, &cx, &cy);
    float dx = static_cast<float>(cx - last_x);
    float dy = static_cast<float>(cy - last_y);
    last_x = cx;
    last_y = cy;

    const float sens = 0.002f;
    cam.yaw += dx * sens;
    cam.pitch -= dy * sens;
    cam.pitch = std::clamp(cam.pitch, -glm::half_pi<float>() + 0.01f,
                           glm::half_pi<float>() - 0.01f);

    glm::vec3 move{0.0f};
    if (glfwGetKey(glfw_win, GLFW_KEY_W) == GLFW_PRESS)
      move += cam.forward();
    if (glfwGetKey(glfw_win, GLFW_KEY_S) == GLFW_PRESS)
      move -= cam.forward();
    if (glfwGetKey(glfw_win, GLFW_KEY_A) == GLFW_PRESS)
      move -= cam.right();
    if (glfwGetKey(glfw_win, GLFW_KEY_D) == GLFW_PRESS)
      move += cam.right();
    if (glfwGetKey(glfw_win, GLFW_KEY_SPACE) == GLFW_PRESS)
      move += glm::vec3(0.0f, 1.0f, 0.0f);
    if (glfwGetKey(glfw_win, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
      move -= glm::vec3(0.0f, 1.0f, 0.0f);
    if (glm::length(move) > 0.0f) {
      cam.position += glm::normalize(move) * (2.0f * dt);
    }

    // Voxel editing hotkeys
    if (glfwGetKey(glfw_win, GLFW_KEY_1) == GLFW_PRESS)
      vox_mode = MODE_CLEAR;
    if (glfwGetKey(glfw_win, GLFW_KEY_2) == GLFW_PRESS)
      vox_mode = MODE_FILL_BOX;
    if (glfwGetKey(glfw_win, GLFW_KEY_3) == GLFW_PRESS)
      vox_mode = MODE_FILL_SPHERE;
    if (glfwGetKey(glfw_win, GLFW_KEY_4) == GLFW_PRESS)
      vox_mode = MODE_NOISE;

    if (glfwGetKey(glfw_win, GLFW_KEY_5) == GLFW_PRESS)
      vox_op = OP_REPLACE;
    if (glfwGetKey(glfw_win, GLFW_KEY_6) == GLFW_PRESS)
      vox_op = OP_UNION;
    if (glfwGetKey(glfw_win, GLFW_KEY_7) == GLFW_PRESS)
      vox_op = OP_INTERSECTION;
    if (glfwGetKey(glfw_win, GLFW_KEY_8) == GLFW_PRESS)
      vox_op = OP_SUBTRACT;

    float spd = 30.0f * dt;
    if (vox_mode == MODE_FILL_SPHERE) {
      if (glfwGetKey(glfw_win, GLFW_KEY_LEFT) == GLFW_PRESS)
        sphere_center.x -= spd;
      if (glfwGetKey(glfw_win, GLFW_KEY_RIGHT) == GLFW_PRESS)
        sphere_center.x += spd;
      if (glfwGetKey(glfw_win, GLFW_KEY_UP) == GLFW_PRESS)
        sphere_center.y += spd;
      if (glfwGetKey(glfw_win, GLFW_KEY_DOWN) == GLFW_PRESS)
        sphere_center.y -= spd;
      if (glfwGetKey(glfw_win, GLFW_KEY_PAGE_UP) == GLFW_PRESS)
        sphere_center.z += spd;
      if (glfwGetKey(glfw_win, GLFW_KEY_PAGE_DOWN) == GLFW_PRESS)
        sphere_center.z -= spd;
      if (glfwGetKey(glfw_win, GLFW_KEY_EQUAL) == GLFW_PRESS)
        sphere_radius += spd;
      if (glfwGetKey(glfw_win, GLFW_KEY_MINUS) == GLFW_PRESS)
        sphere_radius = std::max(1.0f, sphere_radius - spd);
    }
    if (vox_mode == MODE_FILL_BOX) {
      if (glfwGetKey(glfw_win, GLFW_KEY_LEFT) == GLFW_PRESS)
        box_center.x -= spd;
      if (glfwGetKey(glfw_win, GLFW_KEY_RIGHT) == GLFW_PRESS)
        box_center.x += spd;
      if (glfwGetKey(glfw_win, GLFW_KEY_UP) == GLFW_PRESS)
        box_center.y += spd;
      if (glfwGetKey(glfw_win, GLFW_KEY_DOWN) == GLFW_PRESS)
        box_center.y -= spd;
      if (glfwGetKey(glfw_win, GLFW_KEY_PAGE_UP) == GLFW_PRESS)
        box_center.z += spd;
      if (glfwGetKey(glfw_win, GLFW_KEY_PAGE_DOWN) == GLFW_PRESS)
        box_center.z -= spd;
      if (glfwGetKey(glfw_win, GLFW_KEY_EQUAL) == GLFW_PRESS)
        box_half += glm::vec3(spd);
      if (glfwGetKey(glfw_win, GLFW_KEY_MINUS) == GLFW_PRESS)
        box_half = glm::max(box_half - glm::vec3(spd), glm::vec3(1.0f));
    }

    if (glfwGetKey(glfw_win, GLFW_KEY_LEFT_BRACKET) == GLFW_PRESS)
      noise_seed--;
    if (glfwGetKey(glfw_win, GLFW_KEY_RIGHT_BRACKET) == GLFW_PRESS)
      noise_seed++;

    auto fb = window->framebuffer_size();
    VkExtent2D want{static_cast<uint32_t>(fb.first),
                    static_cast<uint32_t>(fb.second)};
    if (want.width == 0 || want.height == 0) {
      std::this_thread::sleep_for(10ms);
      continue;
    }

    if (want.width != last.width || want.height != last.height) {
      vkDeviceWaitIdle(device.device());
      destroy_swapchain_stack();
      create_swapchain_stack(want.width, want.height);
      last = swapchain->extent();
      std::this_thread::sleep_for(1ms);
      continue;
    }

    float rwf = static_cast<float>(swapchain->extent().width) * kRenderScale;
    float rhf = static_cast<float>(swapchain->extent().height) * kRenderScale;
    float jx = halton(jitter_index & 1023u, 2) - 0.5f;
    float jy = halton(jitter_index & 1023u, 3) - 0.5f;
    jitter_index++;
    glm::mat4 view = glm::lookAt(cam.position, cam.position + cam.forward(),
                                 {0.0f, 1.0f, 0.0f});
    glm::mat4 proj =
        glm::perspective(glm::radians(90.0f), rwf / rhf, 0.1f, 100.0f);
    proj[2][0] += jx * 2.0f / rwf;
    proj[2][1] += jy * 2.0f / rhf;
    glm::mat4 view_proj = proj * view;

    CameraUBO ubo{};
    ubo.inv_view_proj = glm::inverse(view_proj);
    ubo.render_resolution = {rwf, rhf};
    ubo.output_resolution = {static_cast<float>(swapchain->extent().width),
                             static_cast<float>(swapchain->extent().height)};
    ubo.time = total_time;
    ubo.debug_normals =
        (glfwGetKey(glfw_win, GLFW_KEY_N) == GLFW_PRESS) ? 1.0f : 0.0f;
    ubo.debug_level =
        (glfwGetKey(glfw_win, GLFW_KEY_L) == GLFW_PRESS) ? 1.0f : 0.0f;
    ubo.debug_steps =
        (glfwGetKey(glfw_win, GLFW_KEY_H) == GLFW_PRESS) ? 1.0f : 0.0f;
    upload_buffer(allocator.raw(), transfer, device.graphics_queue(), cam_buf,
                  &ubo, sizeof(ubo));

    VoxParams vparams{};
    vparams.dim = {static_cast<int>(brick_mgr.brick_size),
                   static_cast<int>(brick_mgr.brick_size),
                   static_cast<int>(brick_mgr.brick_size)};
    vparams.frame = frame_counter++;
    vparams.volMin = {0.0f, 0.0f, 0.0f};
    vparams.volMax = {static_cast<float>(N), static_cast<float>(N),
                      static_cast<float>(N)};
    vparams.boxCenter = box_center;
    vparams.boxHalf = box_half;
    vparams.sphereCenter = sphere_center;
    vparams.sphereRadius = sphere_radius;
    vparams.mode = vox_mode;
    vparams.op = vox_op;
    vparams.noiseSeed = noise_seed;
    vparams.material = 1;
    vparams.terrainFreq = 0.05f;
    vparams.grassDensity = 0.5f;
    vparams.treeDensity = 0.05f;
    vparams.flowerDensity = 0.2f;

    BrickManager::BrickKey cam_key{
        static_cast<int>(std::floor(cam.position.x / brick_mgr.brick_size)),
        static_cast<int>(std::floor(cam.position.y / brick_mgr.brick_size)),
        static_cast<int>(std::floor(cam.position.z / brick_mgr.brick_size))};
    brick_mgr.acquire(cam_key, vparams.frame);
    brick_mgr.stream(cam.position, vparams.frame, 2);
    upload_buffer(allocator.raw(), transfer, device.graphics_queue(),
                  vox_params_buf, &vparams, sizeof(vparams));

    ctx.dispatch_dim = {brick_mgr.brick_size, brick_mgr.brick_size,
                        brick_mgr.brick_size};
    ctx.brick_params.clear();
    for (const auto &kv : brick_mgr.bricks) {
      BrickParams bp{};
      bp.brickCoord = {kv.first.x, kv.first.y, kv.first.z};
      bp.brickIndex = static_cast<int>(kv.second.index);
      ctx.brick_params.push_back(bp);
    }

    commands->acquire_record_present(
        swapchain->vk(), const_cast<VkImage *>(swapchain->images().data()),
        const_cast<VkImageView *>(swapchain->image_views().data()),
        swapchain->image_format(), swapchain->extent(), device.graphics_queue(),
        device.present_queue(), &record_present, &ctx);
    vkQueueWaitIdle(device.graphics_queue());
    void *mapped = nullptr;
    vmaMapMemory(allocator.raw(), steps_buf.allocation, &mapped);
    uint32_t *stepData = static_cast<uint32_t *>(mapped);
    uint64_t sum = 0;
    uint32_t maxv = 0;
    uint32_t cells = ctx.steps_dim.width * ctx.steps_dim.height;
    for (uint32_t i = 0; i < cells; i++) {
      sum += stepData[i];
      if (stepData[i] > maxv)
        maxv = stepData[i];
    }
    vmaUnmapMemory(allocator.raw(), steps_buf.allocation);
    float avg_steps = static_cast<float>(sum) /
                      (static_cast<float>(swapchain->extent().width) *
                       static_cast<float>(swapchain->extent().height));
    float max_steps =
        static_cast<float>(maxv) / static_cast<float>(kStepsDown * kStepsDown);
    spdlog::info("avg_steps {:.2f} max_steps {:.2f}", avg_steps, max_steps);

    std::this_thread::sleep_for(1ms);
  }

  vkDeviceWaitIdle(device.device());

  destroy_swapchain_stack();
  ray_pipeline.reset();
  present_pipeline.reset();

  occ_l1_storage_view.reset();
  occ_l1_view.reset();
  occ_l2_storage_view.reset();
  occ_l2_view.reset();

  brick_mgr.destroy();

  destroy_image3d(allocator.raw(), occ_l1_img);
  destroy_image3d(allocator.raw(), occ_l2_img);
  destroy_buffer(allocator.raw(), cam_buf);
  destroy_buffer(allocator.raw(), vox_buf);
  destroy_buffer(allocator.raw(), vox_params_buf);
    nearest_sampler.reset();
    linear_sampler.reset();
  }
  allocator.destroy();

  spdlog::info("Shutdown.");
  return 0;
}

} // namespace engine
