cmake_minimum_required(VERSION 3.10)
project(xtorchProject)

# Use C++20 (c++2a)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fconcepts")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -fconcepts")

# Export compile commands (generates compile_commands.json)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Manually specify include directories
include_directories(
    /home/bradley/miniconda3/envs/xtensor/include/
    /home/bradley/miniconda3/envs/xtensor/include/xtensor/
    /home/bradley/miniconda3/envs/xtensor/include/xtensor-blas/
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Manually specify library directories
link_directories(/home/bradley/miniconda3/envs/xtensor/lib)

# Build the main executable from main.cpp
add_executable(main main.cpp)

# Link required libraries
target_link_libraries(main OpenImageIO cblas)
