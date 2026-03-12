/*
 * Minimal logging macros to replace ROS logging.
 */

#pragma once

#include <cstdio>

inline bool& ah_verbose_enabled() {
  static bool enabled = false;
  return enabled;
}

inline void ah_set_verbose(bool enabled) {
  ah_verbose_enabled() = enabled;
}

#define AH_LOG_INFO(fmt, ...) \
  do { \
    if (ah_verbose_enabled()) std::fprintf(stderr, "[I] " fmt "\n", ##__VA_ARGS__); \
  } while (0)
#define AH_LOG_WARN(fmt, ...) \
  do { \
    if (ah_verbose_enabled()) std::fprintf(stderr, "[W] " fmt "\n", ##__VA_ARGS__); \
  } while (0)
#define AH_LOG_ERROR(fmt, ...) std::fprintf(stderr, "[E] " fmt "\n", ##__VA_ARGS__)
