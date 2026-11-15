cmake_minimum_required(VERSION 3.22.1)
project(bench C CXX)

# llama.cpp b11836 (MIT), CPU backend only, same options as experiments/android_bench.sh.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)  # phones don't ship libomp.so
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_CPU_KLEIDIAI OFF CACHE BOOL "" FORCE)  # ON would download KleidiAI at build time
foreach(opt LLAMA_BUILD_COMMON LLAMA_BUILD_TESTS LLAMA_BUILD_EXAMPLES LLAMA_BUILD_TOOLS LLAMA_BUILD_SERVER LLAMA_CURL LLAMA_OPENSSL)
    set(${opt} OFF CACHE BOOL "" FORCE)
endforeach()
add_subdirectory(${LLAMA_DIR} llama)

add_library(bench SHARED bench.cpp)
target_link_libraries(bench llama log)
