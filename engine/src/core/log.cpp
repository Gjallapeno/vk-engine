#include <engine/log.hpp>
#include <engine/config.hpp>
#include <spdlog/spdlog.h>

namespace engine {

static inline const char* build_type_str() {
#ifdef NDEBUG
  return "Release";
#else
  return "Debug";
#endif
}

void init_logging() {
  spdlog::set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");
#ifndef NDEBUG
  spdlog::set_level(spdlog::level::trace);
#else
  spdlog::set_level(spdlog::level::info);
#endif
}

void log_boot_banner(std::string_view where) {
  spdlog::info("=== vk_engine boot [{}] ===", where);
  spdlog::info("Config: validation={} | gpu_markers={} | gpu_timers={}",
               engine::cfg::kValidation, engine::cfg::kGpuMarkers, engine::cfg::kGpuTimers);
#if defined(_MSC_VER)
  spdlog::info("Build: {} | Compiler: MSVC {} | Std: C++20", build_type_str(), _MSC_VER);
#elif defined(__clang__)
  spdlog::info("Build: {} | Compiler: Clang {} | Std: C++20", build_type_str(), __clang_version__);
#elif defined(__GNUC__)
  spdlog::info("Build: {} | Compiler: GCC {} | Std: C++20", build_type_str(), __VERSION__);
#else
  spdlog::info("Build: {} | Compiler: Unknown | Std: C++20", build_type_str());
#endif
}

} // namespace engine
