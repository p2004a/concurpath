cmake_minimum_required(VERSION 3.0.2)

set(PROJECT_NAME concurrpath)
set(BINARY_NAME "path")

project(${PROJECT_NAME} C CXX)

add_definitions(-std=c++14 -O3 -Wall -Wextra)

set(${PROJECT_NAME}_LINK_LIBS )

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

find_package(allegro 5.0 REQUIRED COMPONENTS
  main font ttf dialog primitives image
)
include_directories(${allegro_INCLUDE_DIRS})
list(APPEND ${PROJECT_NAME}_LINK_LIBS ${allegro_LIBRARIES})
list(APPEND ${PROJECT_NAME}_LINK_LIBS pthread)

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    )

set(PROJECT_SRC_FILES
    src/main.cpp
    src/display.cpp
    )

add_executable(${BINARY_NAME} ${PROJECT_SRC_FILES})

target_link_libraries(${BINARY_NAME} ${${PROJECT_NAME}_LINK_LIBS})
