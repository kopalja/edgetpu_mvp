cmake_minimum_required(VERSION 3.10)

# Set the project name
set(project_name "inference")
project(${project_name})
add_executable(${project_name} main.cpp)

# Set optimization
set(CMAKE_CXX_FLAGS_RELEASE "-O0")

# Find packages
find_package(Threads REQUIRED)
find_package(TFLiter COMPONENTS Core EdgeTPUBackend REQUIRED)
find_package(OpenCV REQUIRED)

# Link packages
target_link_libraries(${project_name}
    PUBLIC ${OpenCV_LIBS}
    PUBLIC TFLiter::Core
    PUBLIC TFLiter::EdgeTPUBackend
)
