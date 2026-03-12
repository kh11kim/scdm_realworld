cmake_minimum_required(VERSION 3.14)
project(allegro_hand_control LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_TESTS "Build control test utilities" ON)
option(USE_ZMQ "Build ZMQ-based runner" ON)

if(USE_ZMQ)
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(ZMQ libzmq)
  endif()
  if(NOT ZMQ_FOUND)
    message(WARNING "libzmq not found; allegro_run will be skipped. Install libzmq-dev to enable.")
    set(USE_ZMQ OFF)
  endif()
endif()

find_package(Threads REQUIRED)

# Import driver (guard if already added by a top-level build)
if(NOT TARGET allegro_hand_driver)
  add_subdirectory(../allegro_hand_driver ${CMAKE_BINARY_DIR}/allegro_hand_driver)
endif()

# Import prebuilt BHand
add_library(BHand SHARED IMPORTED GLOBAL)
set_target_properties(BHand PROPERTIES
  IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../bhand/lib/libBHand.so
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/../bhand/include
)

add_library(allegro_hand_control
  src/control_core.cpp
)

target_include_directories(allegro_hand_control
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../allegro_hand_driver/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../bhand/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(allegro_hand_control
  PUBLIC
    allegro_hand_driver
    BHand
  PRIVATE
    Threads::Threads
)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(allegro_hand_control PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(TARGETS allegro_hand_control
  EXPORT allegro_hand_control-targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

install(EXPORT allegro_hand_control-targets
  DESTINATION lib/cmake/allegro_hand_control
  NAMESPACE allegro_hand::
)

if(BUILD_TESTS)
  add_executable(ping_control tests/ping_control.cpp)
  target_link_libraries(ping_control PRIVATE allegro_hand_control)
  target_include_directories(ping_control PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../allegro_hand_driver/include)
  install(TARGETS ping_control RUNTIME DESTINATION bin)
endif()

add_executable(allegro_run src/allegro_run.cpp)
target_link_libraries(allegro_run
  PRIVATE
    allegro_hand_control
    ${ZMQ_LIBRARIES}
    Threads::Threads)
target_include_directories(allegro_run PRIVATE ${ZMQ_INCLUDE_DIRS})
if(USE_ZMQ)
  install(TARGETS allegro_run RUNTIME DESTINATION bin)
else()
  set_target_properties(allegro_run PROPERTIES EXCLUDE_FROM_ALL TRUE EXCLUDE_FROM_DEFAULT_BUILD TRUE)
endif()
