
cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(optixUSDLoader)

find_package(CUDA REQUIRED)
find_package(pxr REQUIRED)

# Set C++ standard required by OpenUSD
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenGL and GLUT (used by standard optixUSDLoader sample window)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

# Example target definition in optixUSDLoader/CMakeLists.txt
add_library( optixUSDLoader_ptx OBJECT optixUSDLoader.cu )

# Instruct NVCC to compile this .cu file strictly to PTX
set_source_files_properties( optixUSDLoader.cu PROPERTIES
    COMPILE_OPTIONS "--ptx"
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIRS}
    ${GLUT_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/3rdparty/glad/include
    ${CMAKE_SOURCE_DIR}/3rdparty/GLFW/include
     /usr/local/optix9.1.0/SDK/support
)

# OptiX SDK include path (adjust if installed elsewhere)
include_directories(/usr/local/NVIDIA-OptiX-SDK/include)

# Compile CUDA .cu file to PTX
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3 -gencode arch=compute_75,code=compute_75)
###cuda_compile(OPTIX_PTX_FILES optixUSDLoader.cu)
###cuda_compile_to_ptx( optixUSDLoader_ptx_files optixUSDLoader.cu )

# Compatibility definition for legacy cuda_compile_to_ptx
if(NOT COMMAND cuda_compile_to_ptx)
  macro(cuda_compile_to_ptx generated_files)
    set(${generated_files})
    foreach(it ${ARGN})
      get_filename_component(file_we ${it} NAME_WE)
      set(ptx_file "${CMAKE_CURRENT_BINARY_DIR}/${file_we}.ptx")
      add_custom_command(
        OUTPUT "${ptx_file}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${it}"
        COMMAND ${CMAKE_CUDA_COMPILER}
                -ptx
                -O3
                -arch=sm_89  # Match your GPU architecture capability (e.g. sm_80, sm_89, sm_90)
                -I/usr/local/optix9.1.0/include
                "${CMAKE_CURRENT_SOURCE_DIR}/${it}"
                -o "${ptx_file}"
        COMMENT "Building PTX ${file_we}.ptx"
      )
      list(APPEND ${generated_files} "${ptx_file}")
    endforeach()
  endmacro()
endif()

add_executable(optixUSDLoader
    optixUSDLoader.cpp
    ${OPTIX_PTX_FILES}
)

target_link_libraries(optixUSDLoader PRIVATE
    ${CUDA_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${GLUT_LIBRARIES}
    usd
    usdShade
    usdGeom
    gf
    tf
    glad
    sutil_7_sdk
    glfw
    OpenGL::GL
    dl
    pthread
)
