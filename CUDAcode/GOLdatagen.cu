#include <GOLdatabase_2.hpp>
#include <d_matrix_2.hpp>

extern "C"{
    void genGOLdata(uint32_t seed, uint32_t sample_quantity, double alive_ratio){
        dataset_id config;
        config.alive_ratio = alive_ratio;
        config.sample_quantity = sample_quantity;
        config.seed = seed;
        GOL_2::generateGameOfLifeData(sample_quantity, alive_ratio, seed, config);
    }
    int getPredict(int* initialPattern){
        d_matrix_ver2::d_matrix_2<int> initP;
        std::vector<int> initP_host(initialPattern, initialPattern+100);
        initP.setHostData(initP_host);
        initP.cpyToDev();
        return GOL_2::simulateAndLabel(initP, 0); //fileId인자는 왜 넣었더라? 까먹음
                                                  //내가 예전에 파일 생성 부분이 이 함수에 합쳐져 있어서 각 파일을 분리해주는 인자가 필요했었는데, 지금은 다른 로직으로 분리됨에 따라 쓸모가 없어짐. 
                                                  //지금은 타 함수의 의존성 문제로 인해 제거하지 못하고 있음. 
    }
    void genGOLdataInOneFile(uint32_t seed, uint32_t sample_quantity, double alive_ratio){
        dataset_id config;
        config.alive_ratio = alive_ratio;
        config.sample_quantity = sample_quantity;
        config.seed = seed;
        GOL_2::generateGameOfLifeDataInOneFile(sample_quantity, alive_ratio, seed, config);
    }
}

