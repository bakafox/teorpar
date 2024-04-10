cmake_minimum_required(VERSION 3.0.0)
project(task2.3 VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

add_executable(task2.3 main.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(task2.3 PUBLIC OpenMP::OpenMP_CXX)
endif()
