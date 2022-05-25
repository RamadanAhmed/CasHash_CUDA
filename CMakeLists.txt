cmake_minimum_required(VERSION 3.18)
project(
  CasHash-CUDA
  VERSION 1.0.0.0
  LANGUAGES C CXX CUDA)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds not allowed. Please make a seperate directory and run cmake from there.")
endif()

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(TestKeyFileReader src/TestKeyFileReader.cpp src/KeyFileReader.cpp)

target_include_directories(TestKeyFileReader PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include)

target_compile_features(TestKeyFileReader PRIVATE cxx_std_17)

set_target_properties(TestKeyFileReader
                      PROPERTIES
                      CUDA_ARCHITECTURES 35 50 72
                      CUDA_RESOLVE_DEVICE_SYMBOLS ON
                      CUDA_HOST_COMPILATION_CPP OFF
                      CUDA_PROPAGATE_HOST_FLAGS OFF
                      CUDA_SEPARABLE_COMPILATION ON
                      CUDA_STANDARD 11)

target_link_libraries(TestKeyFileReader 
                      PRIVATE
                      opencv_features2d
                      opencv_highgui
                      opencv_core
                      CUDA::toolkit
                      CUDA::curand
                      CUDA::cudart_static)

add_executable(
  TestHashConverter
  src/TestHashConverter.cpp
  src/KeyFileReader.cpp
  src/HashConverter.cpp
  src/HashConverter.cu)

target_include_directories(TestHashConverter PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include)

target_compile_features(TestHashConverter PRIVATE cxx_std_17)

set_target_properties(TestHashConverter
                      PROPERTIES
                      CUDA_ARCHITECTURES 35 50 72
                      CUDA_RESOLVE_DEVICE_SYMBOLS ON
                      CUDA_HOST_COMPILATION_CPP OFF
                      CUDA_PROPAGATE_HOST_FLAGS OFF
                      CUDA_SEPARABLE_COMPILATION ON
                      CUDA_STANDARD 11)

target_link_libraries(TestHashConverter 
                      PRIVATE
                      opencv_features2d
                      opencv_highgui
                      opencv_core
                      CUDA::toolkit
                      CUDA::curand
                      CUDA::cudart)

add_executable(
  TestHashMatcher
  src/TestHashMatcher.cpp
  src/KeyFileReader.cpp
  src/HashConverter.cpp
  src/HashConverter.cu
  src/HashMatcher.cpp
  src/HashMatcher.cu)

target_link_libraries(TestHashMatcher 
                      PRIVATE
                      opencv_features2d
                      opencv_highgui
                      opencv_core
                      CUDA::toolkit
                      CUDA::curand
                      CUDA::cudart)
target_include_directories(TestHashMatcher PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include)

target_compile_features(TestHashMatcher PRIVATE cxx_std_17)

set_target_properties(TestHashMatcher
                      PROPERTIES
                      CUDA_ARCHITECTURES 35 50 72
                      CUDA_RESOLVE_DEVICE_SYMBOLS ON
                      CUDA_HOST_COMPILATION_CPP OFF
                      CUDA_PROPAGATE_HOST_FLAGS OFF
                      CUDA_SEPARABLE_COMPILATION ON
                      CUDA_STANDARD 11)

add_executable(
  KeyMatchCUDA
  src/main.cpp
  src/KeyFileReader.cpp
  src/HashConverter.cpp
  src/HashConverter.cu
  src/HashMatcher.cpp
  src/HashMatcher.cu)

target_link_libraries(KeyMatchCUDA 
                      PRIVATE
                      curand
                      opencv_features2d
                      opencv_highgui
                      opencv_core
                      CUDA::toolkit
                      CUDA::curand
                      CUDA::cudart)

target_include_directories(KeyMatchCUDA PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include)

target_compile_features(KeyMatchCUDA PRIVATE cxx_std_17)

set_target_properties(KeyMatchCUDA
                      PROPERTIES
                      CUDA_ARCHITECTURES 35 50 72
                      CUDA_RESOLVE_DEVICE_SYMBOLS ON
                      CUDA_HOST_COMPILATION_CPP OFF
                      CUDA_PROPAGATE_HOST_FLAGS OFF
                      CUDA_SEPARABLE_COMPILATION ON
                      CUDA_STANDARD 11)