#pragma once
#include <iostream>
#include <chrono>
#include <fstream>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <filesystem>
#include <iomanip>
#include <sstream>

struct dataset_id{
    uint32_t seed;
    uint32_t sample_quantity;
    double alive_ratio;
};

inline std::string getDatasetId(dataset_id info){
    return "database-" + std::to_string(info.seed) + "_" +
           std::to_string(info.sample_quantity) + "_" +
           std::to_string(info.alive_ratio);
}

inline std::string getCurrentTimestamp()
{
    auto now = std::chrono::system_clock::now();
    std::time_t t_now = std::chrono::system_clock::to_time_t(now);
    std::tm tm_now = *std::localtime(&t_now);

    std::ostringstream oss;
    oss << std::put_time(&tm_now, "%Y-%m-%d_%H%M%S");
    return oss.str();
}

inline void printProgressBar(int current, int total, std::chrono::steady_clock::time_point startTime, std::string processname) {
    int width = 50;
    float progress = static_cast<float>(current) / total;
    int pos = static_cast<int>(width * progress);
    
    auto elapsed = std::chrono::steady_clock::now() - startTime;
    int elapsedSec = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

    std::cout << "[";
    for (int i = 0; i < width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << "% ";
    std::cout << '[' << processname << ']';
    std::cout << "(경과 시간: " << elapsedSec << " ms)                      \r";
    std::cout.flush();
}

inline std::string findProjectRoot() {
    std::filesystem::path currentPath = std::filesystem::current_path();
    
    // 최대 5단계까지 상위 디렉토리 확인
    for (int i = 0; i < 5; i++) {
        // train_data 폴더나 CMakeLists.txt가 있으면 프로젝트 루트로 간주
        if (std::filesystem::exists(currentPath / "train_data") || 
            std::filesystem::exists(currentPath / "CMakeLists.txt") ||
            std::filesystem::exists(currentPath / ".git")) {
            return currentPath.string();
        }
        
        std::filesystem::path parent = currentPath.parent_path();
        if (parent == currentPath) break; // 루트 디렉토리 도달
        currentPath = parent;
    }
    
    // 찾지 못한 경우 현재 디렉토리 반환
    return std::filesystem::current_path().string();
}


