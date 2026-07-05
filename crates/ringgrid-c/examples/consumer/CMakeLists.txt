# Minimal downstream project that consumes ringgrid via find_package.
#
#   cmake -S . -B build -Dringgrid_DIR=<prefix>/lib/cmake/ringgrid
#   cmake --build build
#   ./build/consumer
cmake_minimum_required(VERSION 3.20)
project(ringgrid_consumer LANGUAGES C CXX)

find_package(ringgrid CONFIG REQUIRED)

add_executable(consumer_c ../smoke.c)
target_link_libraries(consumer_c PRIVATE ringgrid::ringgrid)
set_target_properties(consumer_c PROPERTIES C_STANDARD 11)

add_executable(consumer_cpp ../example.cpp)
target_link_libraries(consumer_cpp PRIVATE ringgrid::ringgrid)
set_target_properties(consumer_cpp PROPERTIES CXX_STANDARD 17)
