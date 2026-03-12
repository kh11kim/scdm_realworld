cmake_minimum_required(VERSION 3.14)
project(allegro_hand_driver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_TESTS "Build test utilities" ON)

find_package(Threads REQUIRED)

add_library(allegro_hand_driver
  src/AllegroHandDrv.cpp
  src/candrv/socket_can.cpp
)

target_include_directories(allegro_hand_driver
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(allegro_hand_driver PRIVATE Threads::Threads)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(allegro_hand_driver PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(TARGETS allegro_hand_driver
  EXPORT allegro_hand_driver-targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  EXPORT allegro_hand_driver-targets
  DESTINATION lib/cmake/allegro_hand_driver
)

if(BUILD_TESTS)
  add_executable(ping_can tests/ping_can.cpp)
  target_link_libraries(ping_can PRIVATE allegro_hand_driver)
  target_include_directories(ping_can PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  install(TARGETS ping_can RUNTIME DESTINATION bin)
endif()
