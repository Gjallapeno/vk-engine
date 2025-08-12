add_library(engine STATIC
  src/engine.cpp
  src/camera.cpp
  src/core/log.cpp
  src/core/vk_checks.cpp
  src/gfx/vulkan_instance.cpp
  src/gfx/vulkan_surface.cpp
  src/gfx/vulkan_device.cpp
  src/gfx/vulkan_swapchain.cpp
  src/gfx/vulkan_commands.cpp
  src/gfx/vulkan_pipeline.cpp
  src/gfx/present_pipeline.cpp
  src/gfx/ray_pipeline.cpp
  src/gfx/utils.cpp
  src/gfx/memory.cpp  # empty TU, utilities are header-only
  src/platform/glfw_window.cpp
)

target_include_directories(engine PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if (MSVC)
  target_compile_options(engine PRIVATE /permissive- /W4 /EHsc)
else()
  target_compile_options(engine PRIVATE -Wall -Wextra -Wpedantic)
endif()

option(ENABLE_VALIDATION "Enable Vulkan validation layers" ON)
option(ENABLE_GPU_MARKERS "Enable GPU debug markers" ON)
option(ENABLE_GPU_TIMERS "Enable GPU timestamp queries" ON)

target_compile_definitions(engine PUBLIC
  $<$<BOOL:${ENABLE_VALIDATION}>:ENGINE_ENABLE_VALIDATION=1>
  $<$<BOOL:${ENABLE_GPU_MARKERS}>:ENGINE_ENABLE_GPU_MARKERS=1>
  $<$<BOOL:${ENABLE_GPU_TIMERS}>:ENGINE_ENABLE_GPU_TIMERS=1>
)

target_compile_definitions(engine PRIVATE
  $<$<CONFIG:Debug>:SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE>
  $<$<NOT:$<CONFIG:Debug>>:SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_INFO>
)

# --- Dependencies
# Vulkan was already found in the root CMakeLists
find_package(VulkanMemoryAllocator CONFIG REQUIRED)  # from vcpkg
find_package(glm CONFIG REQUIRED)

target_link_libraries(engine PRIVATE
  Vulkan::Vulkan
  GPUOpen::VulkanMemoryAllocator
  glm::glm
  glfw
  spdlog::spdlog
)
