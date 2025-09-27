#pragma once
#include <d_matrix_2.hpp>
#include <utility.hpp>
#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <random>
#include <filesystem>
#include <iostream>
#include <chrono>
#include <deque>
#include <cstdlib>
#include <thrust/device_vector.h>

namespace GOL_2 {
    using namespace d_matrix_ver2;

    extern const int BOARDWIDTH;
    extern const int BOARDHEIGHT;
    constexpr int BIT_WIDTH = 8;  // 예: 0~255 범위 표현용

    struct pattern{
        public:
           d_matrix_2<int> pattern;
    };

    //원래 oscillator는 "진동자"를 의미하지만, 여기서는 시간이 지나도 살아있는 셀의 개수가 변하지 않는 패턴을 의미하므로
    //진동자 + 정지자 패턴을 합한 개념으로 사용한다.
    extern const pattern oscillator_three_horizontal; //3개의 셀이 가로로 연결된 패턴
    extern const pattern oscillator_three_vertical; //이건 세로
    extern const pattern oscillator_four;//정사각형 형태.
    extern const pattern oscillator_five_left_up;// 대각선 화살촉 형태, 좌-상향 방향.
    extern const pattern oscillator_five_right_down;// 대각선 화살촉 형태, 우-하향 방향.
    extern const pattern oscillator_five_right_up;// 대각선 화살촉 형태, 우-상향 방향.
    extern const pattern oscillator_five_left_down;// 대각선 화살촉 형태, 좌-하향 방향.
    extern const pattern oscillator_six_horizontal;//가로로 길쭉한 육각형 형태.
    extern const pattern oscillator_six_vertical;//세로로 길쭉한 육각형 형태.

    //글라이더 패턴은 세대가 진행됨에 따라 대각 방향으로 이동하는 패턴이다.
    //일반적으로 쓰이는 글라이더 패턴은 5개의 셀로 구성되며, 세대에 따라 개수가 변하지 않는 특징이 있다.
    extern const pattern glider_left_up; // 좌-상향 방향으로 이동하는 글라이더
    extern const pattern glider_left_down; // 좌-하향 방향으로 이동하는 글라이더
    extern const pattern glider_right_up; // 우-상향 방향으로 이동하는 글라이더
    extern const pattern glider_right_down; // 우-하향 방향으로 이동하는 글라이더

    // Game of Life 다음 세대 계산 커널
    __global__ void nextGenKernel(int* current, int* next, int width, int height);

    // 다음 세대 계산 함수
    d_matrix_2<int> nextGen(const d_matrix_2<int>& current, cudaStream_t str = 0);

    // 살아있는 셀 개수 계산
    int countAlive(const d_matrix_2<int>& mat, cudaStream_t str = 0);

    // 고정 비율 패턴 생성 (패딩 포함)
    d_matrix_2<int> generateFixedRatioPatternWithPadding(
        int fullHeight, int fullWidth, 
        int patternHeight, int patternWidth, 
        double aliveRatio, cudaStream_t str = 0
    );

    // Game of Life 데이터 생성 (동일 시드 -> 동일 데이터셋)
    void generateGameOfLifeData(int filenum, double ratio, int seed, dataset_id info);

    void generateGameOfLifeDataInOneFile(int filenum, double ratio, int seed, dataset_id info);

    // 시뮬레이션 및 라벨링 (최종 패턴 반환)
    d_matrix_2<int> simulateAndLabelingtopattern(const d_matrix_2<int>& initialPattern, int fileId, cudaStream_t str = 0);

    // 시뮬레이션 및 라벨링 (살아있는 셀 개수 반환)
    int simulateAndLabel(const d_matrix_2<int>& initialPattern, int fileId, cudaStream_t str = 0);

    // 최적화: 단일 시뮬레이션으로 패턴과 라벨을 동시에 반환
    std::pair<d_matrix_2<int>, int> simulateAndGetBoth(const d_matrix_2<int>& initialPattern, int fileId, cudaStream_t str = 0);

    // 패턴 배치 커널
    __global__ void placePatternKernel(
        int* board, int* pattern, 
        int fullHeight, int fullWidth,
        int patternHeight, int patternWidth,
        int startRow, int startCol
    );

    // 살아있는 셀 카운트 커널
    __global__ void countAliveKernel(int* mat, int* partialSums, int totalSize);
}
