cmake_minimum_required(VERSION 3.14)
project(bhand LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(BHand SHARED IMPORTED)
set_target_properties(BHand PROPERTIES
    IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/lib/libBHand.so
    INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/include/bhand
)

install(DIRECTORY include/bhand/
  DESTINATION include/bhand
  FILES_MATCHING PATTERN "*.h"
)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/lib/libBHand.so
  DESTINATION lib
)

# For consumers, link to imported target BHand. Installation is optional; headers
# and the shared object are installed explicitly above.
