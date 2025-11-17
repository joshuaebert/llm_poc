cmake_minimum_required(VERSION 3.28)
project(llm_poc)

set(LLAMA_USE_SYSTEM_GGML OFF)
set(CMAKE_CXX_STANDARD 26)

add_subdirectory(vendor)
add_subdirectory(src)
