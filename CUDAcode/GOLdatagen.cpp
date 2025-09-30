// GOLdatagen.cpp - Python ctypes 인터페이스용 C 래퍼 함수들
// CPU 전용 빌드 환경을 위한 래퍼 라이브러리

#include "GOLdatabase_host.hpp"
#include <cstdint>
#include <iostream>

// ============================================================================
// C 인터페이스 함수들 (Python ctypes 호출용)
// ============================================================================

extern "C" {

// CPU 전용 데이터 생성 (호스트 메모리 최적화)
void genGOLdataInHost(uint32_t seed, uint32_t sample_quantity, double alive_ratio) {
    std::cout << "🔧 CPU 모드로 데이터 생성 중..." << std::endl;
    
    dataset_id config;
    config.seed = seed;
    config.sample_quantity = sample_quantity;
    config.alive_ratio = static_cast<float>(alive_ratio);
    
    GOL_2_H::generateGameOfLifeDataInHost(config);
}

// CPU 전용 단일 파일 데이터 생성 (CPU 호스트 함수 활용)
void genGOLdataInOneFile(uint32_t seed, uint32_t sample_quantity, double alive_ratio) {
    std::cout << "🔧 CPU 모드로 데이터 생성 중..." << std::endl;
    
    dataset_id config;
    config.seed = seed;
    config.sample_quantity = sample_quantity;
    config.alive_ratio = static_cast<float>(alive_ratio);
    
    GOL_2_H::generateGameOfLifeDataInHost(config);
    std::cout << "✅ CPU 데이터 생성 완료!" << std::endl;
}

// 패턴 예측 함수 (CPU 버전)
int getPredict(int* initialPattern) {
    // CPU 버전에서는 직접 벡터로 변환하여 시뮬레이션
    std::vector<int> pattern(initialPattern, initialPattern + (GOL_2_H::HEIGHT * GOL_2_H::WIDTH));
    return GOL_2_H::simulateAndLabel(pattern);
}

} // extern "C"