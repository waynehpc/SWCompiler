function(PROTOBUF_GENERATE_CPP SRCS HDRS)
  if(NOT ARGN)
    message(
      SEND_ERROR "Error: PROTOBUF_GENERATE_CPP() called without any proto files"
    )
    return()
  endif()

  set(${SRCS})
  set(${HDRS})
  foreach(FIL ${ARGN})
    set(ABS_FIL ${CMAKE_CURRENT_SOURCE_DIR}/${FIL})
    get_filename_component(FIL_WE ${FIL} NAME_WE)
    get_filename_component(FIL_DIR ${ABS_FIL} PATH)

    list(APPEND ${SRCS} "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.cc")
    list(APPEND ${HDRS} "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.h")

    message(
      STATUS
        "FIL:${FIL}; FIL_WE:${FIL_WE} FIL_DIR:${FIL_DIR} ABS_FIL:${ABS_FIL}")
    message(STATUS "CMAKE_CURRENT_BINARY_DIR:${CMAKE_CURRENT_BINARY_DIR}")
    message(STATUS "PROTOBUF_PROTOC_EXECUTABLE:${PROTOBUF_PROTOC_EXECUTABLE}")
    message(STATUS "PROTOBUF_INCLUDE_DIRS:${PROTOBUF_INCLUDE_DIRS}")

    add_custom_command(
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.cc"
             "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.h"
      COMMAND
        ${PROTOBUF_PROTOC_EXECUTABLE} ARGS --cpp_out ${CMAKE_CURRENT_BINARY_DIR}
        -I ${FIL_DIR} ${ABS_FIL} -I ${PROTOBUF_INCLUDE_DIRS}
      DEPENDS ${ABS_FIL} libprotobuf
      COMMENT "Running C++ protocol buffer compiler on ${FIL}"
      VERBATIM)
  endforeach()

  set_source_files_properties(${${SRCS}} ${${HDRS}} PROPERTIES GENERATED TRUE)
  set(${SRCS}
      ${${SRCS}}
      PARENT_SCOPE)
  set(${HDRS}
      ${${HDRS}}
      PARENT_SCOPE)
endfunction()

# ------------------------------------------------------------------------------

if(USE_SYSTEM_LIB)
  message(STATUS "use system lib")
  find_program(
    Protobuf_PROTOC_EXECUTABLE
    NAMES protoc
    DOC "The Google Protocol Buffers Compiler")

  if(Protobuf_PROTOC_EXECUTABLE)
    get_filename_component(_PROTOBUF_INSTALL_PREFIX
                           ${Protobuf_PROTOC_EXECUTABLE} DIRECTORY)
    get_filename_component(_PROTOBUF_INSTALL_PREFIX
                           ${_PROTOBUF_INSTALL_PREFIX}/.. REALPATH)

    message(STATUS "_PROTOBUF_INSTALL_PREFIX: ${_PROTOBUF_INSTALL_PREFIX}")

    find_library(
      Protobuf_PROTOC_LIBRARY
      NAMES protoc
      PATHS ${_PROTOBUF_INSTALL_PREFIX}/lib
      NO_DEFAULT_PATH)
    # message(STATUS "find_library Protobuf_PROTOC_LIBRARY
    # ${Protobuf_PROTOC_LIBRARY}")

    find_library(
      Protobuf_LIBRARY 
      NAMES protobuf 
      PATHS ${_PROTOBUF_INSTALL_PREFIX}/lib
      NO_DEFAULT_PATH)
    #message(STATUS "find_library Protobuf_LIBRARY ${Protobuf_LIBRARY}")

    find_path(
      Protobuf_INCLUDE_DIR google/protobuf/service.h
      PATHS ${_PROTOBUF_INSTALL_PREFIX}/include
      NO_DEFAULT_PATH)
    # message(STATUS "find_path Protobuf_INCLUDE_DIR ${Protobuf_INCLUDE_DIR}")

    find_package(Protobuf REQUIRED)

    # find_package(Protobuf REQUIRED )

    if(Protobuf_FOUND)
      get_filename_component(Protobuf_ROOT ${Protobuf_INCLUDE_DIR} DIRECTORY)
      set(PROTOBUF_ROOT ${Protobuf_ROOT})

      message(STATUS "System Protobuf_INCLUDE_DIRS ${Protobuf_INCLUDE_DIRS}")
      message(STATUS "System Protobuf_LIBRARIES ${Protobuf_LIBRARIES}")
      message(STATUS "System Protobuf_ROOT ${Protobuf_ROOT}")

      set(PROTOBUF_PROTOC_EXECUTABLE ${Protobuf_PROTOC_EXECUTABLE})
      set(PROTOBUF_INCLUDE_DIRS ${Protobuf_INCLUDE_DIRS})

      # add_library(libprotobuf INTERFACE) target_link_libraries(libprotobuf
      # INTERFACE ${Protobuf_LIBRARIES}) target_include_directories(libprotobuf
      # INTERFACE ${Protobuf_INCLUDE_DIRS})

      add_library(libprotobuf STATIC IMPORTED GLOBAL)
      set_target_properties(
        libprotobuf
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${PROTOBUF_ROOT}/include
                   IMPORTED_LOCATION ${PROTOBUF_ROOT}/lib/libprotobuf.a)

      return()
    endif()
  endif()
endif()

# ------------------------------------------------------------------------------
# Enable ExternalProject CMake module
include(ExternalProject)

# set(PROTOBUF_DIR ${PROJECT_SOURCE_DIR}/third_party/protobuf)
set(PROTOBUF_DIR ${SW_BINARY_DIR}/.mslib/protobuf)
set(PROTOBUF_BUILD_DIR ${EXTERNAL_PROJECTS_ROOT}/protobuf)

if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  set(PROTOBUF_LIB ${PROTOBUF_BUILD_DIR}/lib/libprotobufd.a)
else()
  set(PROTOBUF_LIB ${PROTOBUF_BUILD_DIR}/lib/libprotobuf.a)
endif()
set(PROTOBUF_PROTOC_EXECUTABLE ${PROTOBUF_BUILD_DIR}/bin/protoc)

message(STATUS "PROTOBUF_DIR:${PROTOBUF_DIR} PROTOBUF_LIB: ${PROTOBUF_LIB}")
message(STATUS "PROTOBUF_BUILD_DIR:${PROTOBUF_BUILD_DIR}")

ExternalProject_Add(
  ext_protobuf
  SOURCE_DIR ${PROTOBUF_DIR}/cmake
  PREFIX ${PROTOBUF_BUILD_DIR}
  CMAKE_ARGS -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
             -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
             -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
             -DCMAKE_INSTALL_PREFIX=${PROTOBUF_BUILD_DIR}
             -Dprotobuf_BUILD_EXAMPLES=OFF
             -Dprotobuf_BUILD_TESTS=OFF
             -DBUILD_SHARED_LIBS=OFF
             -DCMAKE_POSITION_INDEPENDENT_CODE=ON
  BUILD_BYPRODUCTS ${PROTOBUF_LIB} ${PROTOBUF_PROTOC_EXECUTABLE})

set(PROTOBUF_INC ${PROTOBUF_BUILD_DIR}/include)
file(MAKE_DIRECTORY ${PROTOBUF_INC})

# -----------------------------------------------------------------------------

# ExternalProject_Get_Property(ext_protobuf SOURCE_DIR BINARY_DIR)

# -----------------------------------------------------------------------------

add_library(libprotobuf STATIC IMPORTED GLOBAL)
add_dependencies(libprotobuf ext_protobuf)
set_target_properties(
  libprotobuf
  PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${PROTOBUF_BUILD_DIR}/include
             IMPORTED_LOCATION ${PROTOBUF_LIB})

add_executable(protoc IMPORTED GLOBAL)
add_dependencies(protoc ext_protobuf)
set_target_properties(protoc PROPERTIES IMPORTED_LOCATION
                                        ${PROTOBUF_BUILD_DIR}/bin/protoc)

set(PROTOBUF_ROOT ${PROTOBUF_BUILD_DIR})
set(PROTOBUF_PROTOC_EXECUTABLE protoc)
set(PROTOBUF_INCLUDE_DIRS ${PROTOBUF_BUILD_DIR}/include)
