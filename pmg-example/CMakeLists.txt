cmake_minimum_required(VERSION 3.20)

if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColourReset "${Esc}[m")
  set(ColourBold  "${Esc}[1m")
  set(Red         "${Esc}[31m")
  set(Green       "${Esc}[32m")
  set(Yellow      "${Esc}[33m")
  set(Blue        "${Esc}[34m")
  set(Magenta     "${Esc}[35m")
  set(Cyan        "${Esc}[36m")
  set(White       "${Esc}[37m")
  set(BoldRed     "${Esc}[1;31m")
  set(BoldGreen   "${Esc}[1;32m")
  set(BoldYellow  "${Esc}[1;33m")
  set(BoldBlue    "${Esc}[1;34m")
  set(BoldMagenta "${Esc}[1;35m")
  set(BoldCyan    "${Esc}[1;36m")
  set(BoldWhite   "${Esc}[1;37m")
endif()

option(cpu "Build for CPUs" off)
option(nvidia "Build for NVidia GPUs" off)
option(amd "Build for AMD GPUs" off)

if(cpu AND (nvidia OR amd))
  message(FATAL_ERROR "You cannot select both a CPU build and a GPU (AMD or NVidia) build. Please select only one: cpu, nvidia, amd.")
endif()
if(nvidia AND amd)
  message(FATAL_ERROR "You cannot select both AMD and NVidia GPU builds. Please select only one: cpu, nvidia, amd.")
endif()



set(PROJECT_NAME pmg)
if(amd)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES gfx90a)
  endif()
  project(${PROJECT_NAME} C CXX HIP)
elseif(nvidia)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 90)
  endif()
  project(${PROJECT_NAME} C CXX CUDA)
endif()


if(cpu)
  message(STATUS "${Green}Compiling for CPUs${ColourReset}")
elseif(nvidia)
  message(STATUS "${Green}Compiling for NVIDIA GPUs${ColourReset}")
elseif(amd)
  message(STATUS "${Green}Compiling for AMD GPUs${ColourReset}")
else()
  message(FATAL_ERROR "No target selected for the build. Please select one of the three options: cpu, nvidia, amd")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-Ofast -march=native ${CMAKE_CXX_FLAGS} -DNDEBUG -Wall")
set(CMAKE_C_FLAGS "-Ofast -march=native ${CMAKE_C_FLAGS} -DNDEBUG -Wall")
set(CMAKE_HIP_FLAGS "-Ofast -march=native -stdlib=libstdc++ --gcc-toolchain=/opt/cray/pe/gcc/12.2.0/snos ${CMAKE_CXX_FLAGS} -DNDEBUG -Wall -I/opt/cray/pe/mpich/8.1.29/ofi/gnu/12.3/include")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++20 -O3 -DNDEBUG --extended-lambda --expt-relaxed-constexpr -lineinfo --allow-unsupported-compiler")


# ------------------------------------------------------------------------------
# General configuration
find_package(MPI)
find_package(DOLFINX REQUIRED)
find_package(Boost 1.70 REQUIRED program_options)
# find_package(spdlog REQUIRED)

set(CMAKE_INCLUDE_CURRENT_DIR ON)


# Set the GPU to compile for
if(amd)
  set(GPU_TARGETS "gfx90a" CACHE STRING "GPU targets to compile for")
endif()

if(nvidia)
  set(GPU_TARGETS "Ampere" CACHE STRING "GPU targets to compile for")
endif()

add_custom_command(
  OUTPUT poisson.c
  COMMAND ffcx --sum_factorization ${CMAKE_CURRENT_SOURCE_DIR}/poisson.py
  VERBATIM
  DEPENDS poisson.py
  COMMENT "Compile poisson.py using FFCx"
)

if (amd)
  # If using ROCm and HIP
  # On ROCm rocThrust requires rocPRIM
  find_package(hip REQUIRED)
  find_package(rocprim REQUIRED CONFIG)
  find_package(rocthrust REQUIRED CONFIG)
endif()

if(amd)
  add_definitions(-DUSE_HIP=1)
  set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
elseif(nvidia)
  add_definitions(-DUSE_CUDA=1)
  set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
elseif(cpu)
  add_definitions(-DCPU_ONLY=1)
endif()

add_executable(${PROJECT_NAME} main.cpp ${CMAKE_CURRENT_BINARY_DIR}/poisson.c)

include_directories("../../")

target_link_libraries(${PROJECT_NAME} dolfinx Boost::program_options)

if(amd)
  target_link_libraries(${PROJECT_NAME} roc::rocthrust)
endif()
