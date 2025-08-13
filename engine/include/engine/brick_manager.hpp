#pragma once

#include <engine/gfx/memory.hpp>
#include <engine/gfx/unique_handles.hpp>
#include <glm/glm.hpp>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

namespace engine {

// Manager for streaming 3-D voxel bricks. Each brick is a fixed size volume
// stored inside a larger atlas texture. Bricks are addressed by integer
// coordinates in world space. A separate lookup texture stores the mapping
// from world-space brick coordinates to atlas indices.
class BrickManager {
public:
  struct BrickKey {
    int x = 0, y = 0, z = 0;
    bool operator==(const BrickKey &o) const { return x == o.x && y == o.y && z == o.z; }
  };

  struct BrickKeyHash {
    size_t operator()(const BrickKey &k) const noexcept;
  };

  struct Brick {
    BrickKey key{};
    uint32_t index = 0;
    uint32_t last_used = 0;
  };

  void init(VmaAllocator alloc, VkDevice dev, uint32_t bsize,
            uint32_t maxb, glm::ivec3 map_dim_ = glm::ivec3(64));
  void destroy();

  // Acquire a brick at the given coordinate. If necessary the least recently
  // used brick is recycled.
  uint32_t acquire(const BrickKey &key, uint32_t frame);

  // Remove bricks that are further than 'radius' bricks from the camera.
  void stream(const glm::vec3 &cam_pos, uint32_t frame, int radius);

  VkImage occ_image() const { return occ_atlas.image; }
  VkImage mat_image() const { return mat_atlas.image; }
  VkImage index_image() const { return index_img.image; }

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
};

} // namespace engine

