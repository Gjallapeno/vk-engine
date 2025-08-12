#pragma once
namespace engine::cfg {
#if defined(ENGINE_ENABLE_VALIDATION)
inline constexpr bool kValidation = true;
#else
inline constexpr bool kValidation = false;
#endif

#if defined(ENGINE_ENABLE_GPU_MARKERS)
inline constexpr bool kGpuMarkers = true;
#else
inline constexpr bool kGpuMarkers = false;
#endif

#if defined(ENGINE_ENABLE_GPU_TIMERS)
inline constexpr bool kGpuTimers = true;
#else
inline constexpr bool kGpuTimers = false;
#endif
} // namespace engine::cfg
