cmake_minimum_required(VERSION 3.5)
project(v5lite_trt)

set(CMAKE_CXX_STANDARD 14)

# ---------------------------------------------------------
# 1. CUDA
# ---------------------------------------------------------
find_package(CUDA REQUIRED)
message(STATUS "Find CUDA include at ${CUDA_INCLUDE_DIRS}")
message(STATUS "Find CUDA libraries: ${CUDA_LIBRARIES}")

# ---------------------------------------------------------
# 2. TensorRT (修正部分)
# ---------------------------------------------------------
# 说明：Deb安装会自动把头文件放到 /usr/include/x86_64-linux-gnu/
# 把库文件放到 /usr/lib/x86_64-linux-gnu/
# 我们让 CMake 自动去这些系统路径找
# ---------------------------------------------------------

# 查找头文件 NvInfer.h
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    PATHS /usr/include/x86_64-linux-gnu /usr/include
    DOC "Path to TensorRT headers"
)
message(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")

# 查找库文件 libnvinfer.so
find_library(TENSORRT_LIBRARY_INFER nvinfer
    DOC "Path to nvinfer library"
)

# 查找库文件 libnvonnxparser.so
find_library(TENSORRT_LIBRARY_ONNXPARSER nvonnxparser
    DOC "Path to nvonnxparser library"
)

# 查找库文件 libnvinfer_plugin.so (通常需要这个插件库)
find_library(TENSORRT_LIBRARY_PLUGIN nvinfer_plugin
    DOC "Path to nvinfer_plugin library"
)

# 将找到的库合并到一个变量中
set(TENSORRT_LIBRARY 
    ${TENSORRT_LIBRARY_INFER} 
    ${TENSORRT_LIBRARY_ONNXPARSER}
    ${TENSORRT_LIBRARY_PLUGIN}
)
message(STATUS "Find TensorRT libs: ${TENSORRT_LIBRARY}")

# ---------------------------------------------------------
# 3. OpenCV
# ---------------------------------------------------------
find_package(OpenCV REQUIRED)
message(STATUS "Find OpenCV include at ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Find OpenCV libraries: ${OpenCV_LIBRARIES}")

# ---------------------------------------------------------
# 4. Local Includes
# ---------------------------------------------------------
set(COMMON_INCLUDE ./includes/common)
find_package(yaml-cpp REQUIRED)

# 包含路径
include_directories(
    ${CUDA_INCLUDE_DIRS} 
    ${TENSORRT_INCLUDE_DIR}  # 这里引用上面 find_path 找到的路径
    ${OpenCV_INCLUDE_DIRS} 
    ${COMMON_INCLUDE} 
    ${YAML_CPP_INCLUDE_DIRS}
)

link_directories(${YAML_CPP_LIB_DIR})

# ---------------------------------------------------------
# 5. Build Target
# ---------------------------------------------------------
add_executable(v5lite_trt main.cpp v5lite.cpp)

target_link_libraries(v5lite_trt 
    ${OpenCV_LIBRARIES} 
    ${CUDA_LIBRARIES} 
    ${TENSORRT_LIBRARY}  # 这里引用上面 find_library 找到的库
    yaml-cpp
)
