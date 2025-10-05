#include <fstream>
#include <random>
#include <filesystem>
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <thread>
#include <deque>
#include <numeric>
#include "utility.hpp"

namespace GOL_2_H{
    constexpr int MAXGEN = 2500;
    constexpr int BOARDHEIGHT = 100;
    constexpr int BOARDWIDTH = 100;

    constexpr int HEIGHT = 10;
    constexpr int WIDTH = 10;

    // 메모리 효율적인 버전 - 참조 전달로 메모리 재사용
    void genRandomPattern(std::vector<int>& pattern, int seed, float alive_ratio);
    void nextGen(const std::vector<int>& src, std::vector<int>& dst, int width, int height);
    void padding(const std::vector<int>& pattern, std::vector<int>& board, int full_w, int full_h, int pattern_w, int pattern_h);
    int simulateAndLabel(const std::vector<int>& init_pattern);
    
    void generateGameOfLifeDataInHost(dataset_id data_config);
}


