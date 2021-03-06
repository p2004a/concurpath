cmake_minimum_required(VERSION 3.0.2)

set(PROJECT_NAME concurrpath)
set(BINARY_NAME "path")

project(${PROJECT_NAME} C CXX)

add_definitions(-std=c++14 -O2 -Wall -Wextra)

set(${PROJECT_NAME}_LINK_LIBS )

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

find_package(allegro 5.0 REQUIRED COMPONENTS
  main font ttf dialog primitives image
)
include_directories(${allegro_INCLUDE_DIRS})
list(APPEND ${PROJECT_NAME}_LINK_LIBS ${allegro_LIBRARIES})

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS})
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS})
endif(OPENMP_FOUND)

find_package(Threads REQUIRED)
list(APPEND ${PROJECT_NAME}_LINK_LIBS ${CMAKE_THREAD_LIBS_INIT})

find_package(CUDA REQUIRED)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
LIST(APPEND CUDA_NVCC_FLAGS --optimize 2)

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    )

set(PROJECT_SRC_FILES
    src/main.cpp
    src/display.cpp
    src/map.cpp
    src/fpscounter.cpp
    src/simulation.cu
    src/pathfinder.cpp
    )

cuda_add_executable(${BINARY_NAME} ${PROJECT_SRC_FILES})

target_link_libraries(${BINARY_NAME} ${${PROJECT_NAME}_LINK_LIBS})

file(COPY
    ${CMAKE_CURRENT_SOURCE_DIR}/fonts
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
