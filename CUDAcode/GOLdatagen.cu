#include <GOLdatabase_2.hpp>
#include <d_matrix_2.hpp>
#include <chrono>
#include <iostream>
#include <random>
#include <algorithm>

extern "C"{
    
    // 패턴 예측 함수 (GPU 버전)
    int getPredict(int* initialPattern){
        d_matrix_ver2::d_matrix_2<int> initP(10, 10);  // 명시적으로 10x10 크기 설정
        std::vector<int> initP_host(initialPattern, initialPattern+100);
        initP.setHostData(initP_host);
        initP.cpyToDev();
        return GOL_2::simulateAndLabel(initP, 0); // fileId는 사용되지 않음 (레거시)
    }
    
    // 최적화된 패턴 예측 함수 (GPU 버전) - 배치 처리로 성능 향상
    int getPredictOptimized(int* initialPattern){
        d_matrix_ver2::d_matrix_2<int> initP(10, 10);  // 명시적으로 10x10 크기 설정
        std::vector<int> initP_host(initialPattern, initialPattern+100);
        initP.setHostData(initP_host);
        initP.cpyToDev();
        return GOL_2::simulatePatternInKernal(initP, 0); // 새로운 최적화된 함수 사용
    }
    
    // 실제 조건 성능 비교: 10x10 패턴을 100x100 보드에 배치 + 시뮬레이션
    void benchmarkRealConditions(double alive_ratio, int seed, int iterations) {
        std::cout << "🏁 실제 조건 성능 벤치마크" << std::endl;
        std::cout << "조건: 10x10 패턴 → 100x100 보드, 생존율 " << alive_ratio << ", " << iterations << "회 반복" << std::endl;
        
        // 10x10 패턴 생성 (CPU에서)
        int totalCells = 10 * 10;
        int aliveCells = static_cast<int>(totalCells * alive_ratio);
        std::vector<int> host_pattern(totalCells, 0);
        std::fill_n(host_pattern.begin(), aliveCells, 1);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // 기존 함수 벤치마크 (패턴 배치 + 시뮬레이션)
        for (int i = 0; i < iterations; i++) {
            // 패턴 셔플 (매번 다른 패턴)
            std::mt19937_64 gen(seed + i);
            std::shuffle(host_pattern.begin(), host_pattern.end(), gen);
            
            // GPU 메모리로 패턴 복사
            d_matrix_ver2::d_matrix_2<int> pattern(10, 10);
            pattern.setHostData(host_pattern);
            pattern.cpyToDev();
            
            // 100x100 보드 생성 및 초기화
            d_matrix_ver2::d_matrix_2<int> fullBoard(100, 100);
            fullBoard.fill(0);
            
            // 패턴을 보드 중앙에 배치 (placePatternKernel 사용)
            dim3 blockSize(16, 16);
            dim3 gridSize((10 + 15) / 16, (10 + 15) / 16);
            int startRow = (100 - 10) / 2;
            int startCol = (100 - 10) / 2;
            
            GOL_2::placePatternKernel<<<gridSize, blockSize>>>(
                fullBoard.getDevPointer(), pattern.getDevPointer(),
                100, 100, 10, 10, startRow, startCol);
            cudaDeviceSynchronize();
            
            // 기존 시뮬레이션
            GOL_2::simulateAndLabel(fullBoard, 0);
        }
        
        auto mid = std::chrono::high_resolution_clock::now();
        
        // 최적화된 함수 벤치마크 (패턴 배치 + 시뮬레이션)
        for (int i = 0; i < iterations; i++) {
            // 패턴 셔플 (동일한 시드로 동일한 패턴)
            std::mt19937_64 gen(seed + i);
            std::shuffle(host_pattern.begin(), host_pattern.end(), gen);
            
            // GPU 메모리로 패턴 복사
            d_matrix_ver2::d_matrix_2<int> pattern(10, 10);
            pattern.setHostData(host_pattern);
            pattern.cpyToDev();
            
            // 100x100 보드 생성 및 초기화
            d_matrix_ver2::d_matrix_2<int> fullBoard(100, 100);
            fullBoard.fill(0);
            
            // 패턴을 보드 중앙에 배치 (동일한 커널 사용)
            dim3 blockSize(16, 16);
            dim3 gridSize((10 + 15) / 16, (10 + 15) / 16);
            int startRow = (100 - 10) / 2;
            int startCol = (100 - 10) / 2;
            
            GOL_2::placePatternKernel<<<gridSize, blockSize>>>(
                fullBoard.getDevPointer(), pattern.getDevPointer(),
                100, 100, 10, 10, startRow, startCol);
            cudaDeviceSynchronize();
            
            // 최적화된 시뮬레이션
            GOL_2::simulatePatternInKernal(fullBoard, 0);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        
        auto time1 = std::chrono::duration_cast<std::chrono::microseconds>(mid - start);
        auto time2 = std::chrono::duration_cast<std::chrono::microseconds>(end - mid);
        
        std::cout << "=== 실제 조건 성능 비교 결과 ===" << std::endl;
        std::cout << "기존 함수 (패턴배치+시뮬레이션): " << time1.count() / 1000.0 << " ms" << std::endl;
        std::cout << "최적화 함수 (패턴배치+시뮬레이션): " << time2.count() / 1000.0 << " ms" << std::endl;
        std::cout << "성능 향상: " << (double)time1.count() / time2.count() << "배" << std::endl;
        std::cout << "평균 처리 시간: 기존 " << time1.count() / 1000.0 / iterations << "ms/패턴, ";
        std::cout << "최적화 " << time2.count() / 1000.0 / iterations << "ms/패턴" << std::endl;
    }


    
    // ⚡ CPU-GPU 오버헤드 최적화된 GPU 단일파일 데이터 생성
    void genGOLdataOptimizeInOneFile(uint32_t seed, uint32_t sample_quantity, double alive_ratio){
        std::cout << "📁⚡ OPTIMIZED 단일 파일 모드로 데이터 생성 중..." << std::endl;
        dataset_id config;
        config.alive_ratio = alive_ratio;
        config.sample_quantity = sample_quantity;
        config.seed = seed;
        GOL_2::generateGameOfLifeDataOptimizeInOneFile(sample_quantity, alive_ratio, seed, config);
        std::cout << "✅ OPTIMIZED 단일 파일 데이터 생성 완료!" << std::endl;
    }

    void genGOLdataOptimizeInOneFile_RootFlexible(uint32_t seed, uint32_t sample_quantity, double alive_ratio, const char* root){
        std::cout << "🚀⚡ OPTIMIZED GPU 모드로 데이터 생성 중..." << std::endl;
        dataset_id config;
        config.alive_ratio = alive_ratio;
        config.sample_quantity = sample_quantity;
        config.seed = seed;
        GOL_2::generateGameOfLifeDataOptimizeInOneFile_RootFlexible(sample_quantity, alive_ratio, seed, config, static_cast<std::string>(root));
        std::cout << "✅ OPTIMIZED GPU 데이터 생성 완료!" << std::endl;
    }
}

