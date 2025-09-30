#include <GOLdatabase_2.hpp>
#include <d_matrix_2.hpp>

extern "C"{
    // GPU 멀티파일 데이터 생성
    void genGOLdata(uint32_t seed, uint32_t sample_quantity, double alive_ratio){
        std::cout << "🚀 GPU 모드로 데이터 생성 중..." << std::endl;
        dataset_id config;
        config.alive_ratio = alive_ratio;
        config.sample_quantity = sample_quantity;
        config.seed = seed;
        GOL_2::generateGameOfLifeData(sample_quantity, alive_ratio, seed, config);
        std::cout << "✅ GPU 데이터 생성 완료!" << std::endl;
    }
    
    // GPU 단일파일 데이터 생성
    void genGOLdataInOneFile(uint32_t seed, uint32_t sample_quantity, double alive_ratio){
        std::cout << "📁 단일 파일 모드로 데이터 생성 중..." << std::endl;
        dataset_id config;
        config.alive_ratio = alive_ratio;
        config.sample_quantity = sample_quantity;
        config.seed = seed;
        GOL_2::generateGameOfLifeDataInOneFile(sample_quantity, alive_ratio, seed, config);
        std::cout << "✅ 단일 파일 데이터 생성 완료!" << std::endl;
    }
    
    // 패턴 예측 함수 (GPU 버전)
    int getPredict(int* initialPattern){
        d_matrix_ver2::d_matrix_2<int> initP;
        std::vector<int> initP_host(initialPattern, initialPattern+100);
        initP.setHostData(initP_host);
        initP.cpyToDev();
        return GOL_2::simulateAndLabel(initP, 0); // fileId는 사용되지 않음 (레거시)
    }
}

