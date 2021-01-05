cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)
project(SWCompiler C CXX)

# SWCompiler version
set(SWCOMPILER_VERSION_MAJOR 0)
set(SWCOMPILER_VERSION_MINOR 0)
set(SWCOMPILER_VERSION_ADDIT 0)
message(
  STATUS
    "This is SWCompiler ${SWCOMPILER_VERSION_MAJOR}.${SWCOMPILER_VERSION_MINOR}.${SWCOMPILER_VERSION_ADDIT}!"
)

# Options
option(MKLDNN "Use mkldnn for cpu backend" OFF)
option(USE_SYSTEM_LIB "Build with system libraries." ON)

# Build Type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_BUILD_TYPE STREQUAL Debug)
  add_definitions(-DDEBUG)
endif()

if(LEVELDEBUG)
  if(CMAKE_BUILD_TYPE STREQUAL Debug)
    add_definitions(-DLEVELDEBUG=${LEVELDEBUG})
  else()
    add_definitions(-DLEVELDEBUG=100)
  endif()
endif()

if(MKLDNN)
  add_definitions(-DMKLDNN)
endif()

set(SW_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(SW_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${SW_BINARY_DIR})

# Cmake Modules
set(EXTERNAL_PROJECTS_ROOT ${CMAKE_CURRENT_BINARY_DIR}/external)
include(cmake/Util.cmake)
include(cmake/external_libs/protobuf.cmake)

# list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# Architecture level Optimised level

# Headers
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)

# Generic compilation options Flags
set(COMMON_FLAGS
    "-Wall -Wno-uninitialized -Wno-deprecated-declarations -Wno-unknown-pragmas -Wsign-compare ${ADDITIONAL_FLAGS}"
)
set(CMAKE_C_FLAGS "${COMMON_FLAGS} -fno-strict-aliasing")
set(CMAKE_CXX_FLAGS "${COMMON_FLAGS} -std=c++17")

# Definitions

add_subdirectory(src)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
add_definitions(-DGOOGLE_PROTOBUF_NO_RTTI)

#set(PROTOBUF_INCLUDE_DIRS, "/home/wwz/software/protobuf-3.14.0/build/include")
message(STATUS "hhhh ${PROTOBUF_INCLUDE_DIRS}")
include_directories(${PROTOBUF_INCLUDE_DIRS})

protobuf_generate_cpp(CAFFE2_SRCS CAFFE2_HDRS src/importer/caffe2.proto)
# message(STATUS "CAFFE2_HDRS ${CAFFE2_HDRS}")
# message(STATUS "CAFFE2_SRCS${CAFFE2_SRCS}")

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Source
file(GLOB_RECURSE SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp)

add_library(graphcore STATIC ${SOURCES} ${CAFFE2_SRCS}
                             ${CMAKE_CURRENT_BINARY_DIR}/caffe2.pb.h)
target_compile_definitions(graphcore INTERFACE -DGOOGLE_PROTOBUF_NO_RTTI)
target_link_libraries(graphcore PUBLIC libprotobuf Threads::Threads)

# Libraries are compiled
link_libraries(graphcore)

# now build binaries
add_subdirectory(test)
