add_executable(vk_app main.cpp)
target_link_libraries(vk_app PRIVATE engine)

if (MSVC)
  target_compile_options(vk_app PRIVATE /permissive- /W4 /EHsc)
else()
  target_compile_options(vk_app PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Ensure shaders are built before the app is run/copied
add_dependencies(vk_app shaders)
