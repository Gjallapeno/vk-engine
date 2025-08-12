#pragma once
#include <cstdlib>
#include <vulkan/vulkan.h>
#include <spdlog/spdlog.h>

namespace engine::vk {
const char* to_string(VkResult);
}

// Platform-safe debug trap (resolved BEFORE macro expansion)
#if defined(_MSC_VER) && !defined(NDEBUG)
  #define ENGINE_DEBUG_TRAP() __debugbreak()
#else
  #define ENGINE_DEBUG_TRAP() ((void)0)
#endif

#ifndef VK_CHECK
#define VK_CHECK(expr) do {                                 \
  VkResult _vk = (expr);                                    \
  if (_vk != VK_SUCCESS) {                                  \
    spdlog::error("VK_CHECK failed: {} -> {}",              \
                  #expr, engine::vk::to_string(_vk));       \
    ENGINE_DEBUG_TRAP();                                    \
    std::abort();                                           \
  }                                                         \
} while(0)
#endif
