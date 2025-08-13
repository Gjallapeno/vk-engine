#pragma once
#include <mutex>
#include <cstdint>

namespace engine {

class GlfwContext {
public:
  GlfwContext();
  ~GlfwContext();

private:
  static std::mutex mtx_;
  static uint32_t ref_count_;
};

} // namespace engine

