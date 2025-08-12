cmake_minimum_required(VERSION 3.24)
project(vk_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Vulkan REQUIRED)                # Vulkan SDK (we’ll find glslc below)
find_package(glfw3 CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

add_subdirectory(engine)
add_subdirectory(app)

# -------- Shaders (GLSL -> SPIR-V using glslc from the Vulkan SDK) --------
set(SHADER_SRC_DIR ${CMAKE_SOURCE_DIR}/assets/shaders)
set(SHADER_BIN_DIR ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_BIN_DIR})

# Locate glslc robustly
set(VK_GLSLC "")
if (DEFINED Vulkan_GLSLC_EXECUTABLE)
  set(VK_GLSLC ${Vulkan_GLSLC_EXECUTABLE})
elseif (DEFINED VULKAN_GLSLC_EXECUTABLE)
  set(VK_GLSLC ${VULKAN_GLSLC_EXECUTABLE})
else()
  find_program(VK_GLSLC NAMES glslc
    HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/Bin32")
endif()

if (NOT VK_GLSLC)
  message(FATAL_ERROR "glslc not found. Ensure Vulkan SDK is installed and glslc.exe is in PATH.")
endif()

function(compile_shader out_var src stage)
  get_filename_component(fname ${src} NAME)
  set(out ${SHADER_BIN_DIR}/${fname}.spv)
  add_custom_command(
    OUTPUT ${out}
    COMMAND "${VK_GLSLC}" ${src} -o ${out} -g -O --target-env=vulkan1.3 -fshader-stage=${stage}
    DEPENDS ${src}
    COMMENT "Compiling ${src} -> ${out}"
    VERBATIM
  )
  set(${out_var} ${out} PARENT_SCOPE)
endfunction()

compile_shader(TRI_VS ${SHADER_SRC_DIR}/tri.vert vert)
compile_shader(TRI_FS ${SHADER_SRC_DIR}/tri.frag frag)
compile_shader(TEX_VS ${SHADER_SRC_DIR}/tex.vert vert)
compile_shader(TEX_FS ${SHADER_SRC_DIR}/tex.frag frag)

add_custom_target(shaders ALL DEPENDS ${TRI_VS} ${TRI_FS} ${TEX_VS} ${TEX_FS})

# Copy SPIR-V next to vk_app.exe (e.g., build/app/Debug/shaders)
add_custom_command(TARGET shaders POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:vk_app>/shaders
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${TRI_VS} ${TRI_FS} ${TEX_VS} ${TEX_FS} $<TARGET_FILE_DIR:vk_app>/shaders
)
