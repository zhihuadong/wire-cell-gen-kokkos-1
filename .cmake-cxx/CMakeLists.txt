cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(wire-cell-gen-cxx CXX)

if (NOT ${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" AND
    NOT ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
  message(FATAL_ERROR "Must use either GCC or Clang to build the non-Kokkos components of wire-cell-gen-kokkos.\n"
    "Using ${CMAKE_CXX_COMPILER_ID}")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS Off)

file(GLOB all_files ${PROJECT_SOURCE_DIR}/../src/*.cxx)

include_directories(SYSTEM $ENV{BOOST_INC})

set(BOOST_ROOT $ENV{BOOST_DIR} )
set(BOOST_INCLUDEDIR $ENV{BOOST_INC})
set(BOOST_LIBRARYDIR $ENV{BOOST_LIB})
set(Boost_USE_MULTITHREADED ON)

find_package(Boost REQUIRED COMPONENTS)

string(APPEND CMAKE_CXX_FLAGS "-g -O3 -pedantic -fopenmp -Wall")
string(APPEND CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")

add_library(WireCellGenKokkos_cxx SHARED ${all_files})
target_include_directories(WireCellGenKokkos_cxx
  PRIVATE
    ${PROJECT_SOURCE_DIR}/../inc
    $ENV{EIGEN_INC}
    $ENV{JSONCPP_INC}
    $ENV{JSONNET_INC}
    $ENV{SPDLOG_INC}
    $ENV{WIRECELL_INC}
)

target_link_directories(WireCellGenKokkos_cxx PRIVATE $ENV{JSONCPP_LIB} $ENV{WIRECELL_LIB})
target_link_libraries(WireCellGenKokkos_cxx PRIVATE jsoncpp WireCellIface WireCellUtil Boost::headers)
