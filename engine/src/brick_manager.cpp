#include <engine/brick_manager.hpp>
#include <engine/vk_checks.hpp>
#include <cmath>

namespace engine {

size_t BrickManager::BrickKeyHash::operator()(const BrickKey &k) const noexcept {
  size_t h = std::hash<int>()(k.x);
  h ^= std::hash<int>()(k.y) + 0x9e3779b9 + (h << 6) + (h >> 2);
  h ^= std::hash<int>()(k.z) + 0x9e3779b9 + (h << 6) + (h >> 2);
  return h;
}

void BrickManager::init(VmaAllocator alloc, VkDevice dev, uint32_t bsize,
                        uint32_t maxb, glm::ivec3 map_dim_) {
  allocator = alloc;
  device = dev;
  brick_size = bsize;
  max_bricks = maxb;
  map_dim = map_dim_;

  occ_atlas = create_image3d(allocator, brick_size, brick_size,
                             brick_size * max_bricks, VK_FORMAT_R8_UINT,
                             VK_IMAGE_USAGE_STORAGE_BIT |
                                 VK_IMAGE_USAGE_SAMPLED_BIT);
  mat_atlas = create_image3d(allocator, brick_size, brick_size,
                             brick_size * max_bricks, VK_FORMAT_R8_UINT,
                             VK_IMAGE_USAGE_STORAGE_BIT |
                                 VK_IMAGE_USAGE_SAMPLED_BIT);
  index_img = create_image3d(allocator, map_dim.x, map_dim.y, map_dim.z,
                             VK_FORMAT_R32_UINT,
                             VK_IMAGE_USAGE_STORAGE_BIT |
                                 VK_IMAGE_USAGE_SAMPLED_BIT);

  VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  vi.viewType = VK_IMAGE_VIEW_TYPE_3D;
  vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  vi.subresourceRange.levelCount = 1;
  vi.subresourceRange.layerCount = 1;

  vi.image = occ_atlas.image;
  vi.format = VK_FORMAT_R8_UINT;
  VK_CHECK(vkCreateImageView(device, &vi, nullptr, occ_view.init(device)));
  VK_CHECK(vkCreateImageView(device, &vi, nullptr,
                             occ_storage_view.init(device)));

  vi.image = mat_atlas.image;
  vi.format = VK_FORMAT_R8_UINT;
  VK_CHECK(vkCreateImageView(device, &vi, nullptr, mat_view.init(device)));
  VK_CHECK(vkCreateImageView(device, &vi, nullptr,
                             mat_storage_view.init(device)));

  vi.image = index_img.image;
  vi.format = VK_FORMAT_R32_UINT;
  VK_CHECK(vkCreateImageView(device, &vi, nullptr, index_view.init(device)));
  VK_CHECK(vkCreateImageView(device, &vi, nullptr,
                             index_storage_view.init(device)));

  cpu_index.resize(static_cast<size_t>(map_dim.x) * map_dim.y * map_dim.z,
                   0xFFFFFFFFu);
  free_list.reserve(max_bricks);
  lru.resize(max_bricks);
  for (uint32_t i = 0; i < max_bricks; ++i)
    free_list.push_back(max_bricks - 1 - i);
}

void BrickManager::destroy() {
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

uint32_t BrickManager::acquire(const BrickKey &key, uint32_t frame) {
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

  size_t offset =
      (static_cast<size_t>(key.z) * map_dim.y + key.y) * map_dim.x + key.x;
  if (offset < cpu_index.size()) cpu_index[offset] = slot;
  return slot;
}

void BrickManager::stream(const glm::vec3 &cam_pos, uint32_t frame,
                          int radius) {
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

} // namespace engine

