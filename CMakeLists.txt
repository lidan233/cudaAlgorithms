cmake_minimum_required(VERSION 3.17)
project(cudaSampler LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)

find_package(OpenMP)
IF(OPENMP_FOUND)
    message("Found OpenMP!")
    # add flags for OpenMP
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_SHARED_LINKER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
ELSE()
    message("Missed OpenMP!")
ENDIF()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")


include(cmake/lidanBase.cmake)
include(cmake/LidanBuild.cmake)


Lidan_ADD_SRC(PATH "${CMAKE_CURRENT_SOURCE_DIR}/external/glad/" SOURCEDIR gladsrc)
Lidan_ADD_LIB(PATH "${CMAKE_CURRENT_SOURCE_DIR}/external/glfw/" INCLUDEDIRS includedir MSVC "TRUE" SOURCEDIR glfwsrc COMPONENTS "glfw3")
add_subdirectory(external/cxxopts)
message(STATUS "  shit ----${glfwsrc}" )


include_directories("C:/ProgramData/NVIDIA Corporation/CUDA Samples/v10.0/common/inc"
                    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/include"
                    ${CMAKE_CURRENT_SOURCE_DIR}/common
                    external/cxxopts/include)
link_directories(C:/ProgramData/NVIDIA Corporation/CUDA Samples/v10.0/common/lib/x64)


ADD_SUBSRC(commondir common)


message(STATUS ${gladsrc})
add_executable(cudaSampler
        ${commondir}
        ${glfwsrc}
        ${gladsrc}
#        execute/addsum.cu
        common/shader.h
#        execute/testgpuregister.cu
#        execute/mergesort.cu
#         execute/radix.cu
        execute/addsum.cu
#        execute/mycudamergesort.cu
        )



set_target_properties(
        cudaSampler
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(${PROJECT_NAME} glfw3 )