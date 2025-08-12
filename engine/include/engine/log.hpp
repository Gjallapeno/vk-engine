#pragma once
#include <string_view>
namespace engine {
  void init_logging();
  void log_boot_banner(std::string_view where);
}
