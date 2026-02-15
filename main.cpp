// main.cpp
#include "v5lite.h"
#include <iostream>
#include <string>
#include <sys/stat.h> // 用于判断文件类型

// 辅助函数：判断是否为视频文件
bool isVideoFile(const std::string& path) {
    std::string ext = path.substr(path.find_last_of(".") + 1);
    return (ext == "mp4" || ext == "avi" || ext == "mkv" || ext == "mov");
}

// 辅助函数：判断是否为图片文件
bool isImageFile(const std::string& path) {
    std::string ext = path.substr(path.find_last_of(".") + 1);
    return (ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp");
}

// 辅助函数：判断路径是否存在且是文件夹
bool isFolder(const std::string& path) {
    struct stat s;
    if (stat(path.c_str(), &s) == 0) {
        return s.st_mode & S_IFDIR;
    }
    return false;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: ./yolov5_trt [config_path] [input_path/webui]" << std::endl;
        return -1;
    }

    std::string configPath = argv[1];
    std::string inputPath = argv[2];

    // 1. 初始化模型
    V5lite V5lite(configPath); // 假设构造函数接收 config 路径
    V5lite.LoadEngine();

    // 2. 分发逻辑
    if (inputPath == "webui") {
        // === WebUI 服务模式 (Pipe通信) ===
        // 这是为了支持 Python 前端 "上传一张，推理一张" 且不重新加载模型
        std::cout << "READY" << std::endl; // 握手信号
        
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line == "exit") break;
            if (line.empty()) continue;
            
            // 判断输入是图片还是视频
            if (isVideoFile(line)) {
                std::string resPath = V5lite.InferenceVideo(line);
                std::cout << resPath << std::endl; // 返回结果路径
            } else {
                std::string resPath = V5lite.InferenceImage(line);
                std::cout << resPath << std::endl; // 返回结果路径
            }
        }
    } 
    else if (isFolder(inputPath)) {
        // === 原有的文件夹批量模式 ===
        std::cout << "Mode: Folder Inference" << std::endl;
        V5lite.InferenceFolder(inputPath);
    }
    else if (isVideoFile(inputPath)) {
        // === 单视频模式 ===
        std::cout << "Mode: Single Video Inference" << std::endl;
        V5lite.InferenceVideo(inputPath);
    }
    else {
        // === 单图片模式 (默认) ===
        std::cout << "Mode: Single Image Inference" << std::endl;
        V5lite.InferenceImage(inputPath);
    }

    return 0;
}
