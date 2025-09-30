#include "GOLdatabase_host.hpp"

void GOL_2_H::genRandomPattern(std::vector<int>& pattern, int seed, float alive_ratio)
{
    int totalPatternCells = HEIGHT * WIDTH;
    int aliveCells = static_cast<int>(totalPatternCells * alive_ratio);
    
    // 패턴 벡터 크기 확보 및 초기화
    pattern.resize(totalPatternCells);
    std::fill(pattern.begin(), pattern.end(), 0);
    
    // 살아있는 셀만큼 1로 설정
    std::fill_n(pattern.begin(), aliveCells, 1);
    
    // 셔플로 랜덤 분배
    std::mt19937_64 gen(seed);
    std::shuffle(pattern.begin(), pattern.end(), gen);
}

void GOL_2_H::nextGen(const std::vector<int>& src, std::vector<int>& dst, int width, int height)
{
    // 목표 벡터 크기 확보
    dst.resize(src.size());
    
    for(int y = 0; y < height; y++){
        for(int x = 0; x < width; x++){
            int currentIdx = y * width + x;
            int aliveCount = 0;
            
            // 경계 검사를 최적화한 이웃 계산
            for(int dy = -1; dy <= 1; dy++){
                for(int dx = -1; dx <= 1; dx++){
                    int ny = y + dy;
                    int nx = x + dx;
                    
                    // 경계 검사 최적화
                    if(ny >= 0 && ny < height && nx >= 0 && nx < width) {
                        aliveCount += src[ny * width + nx];
                    }
                }
            }
            
            // Game of Life 규칙 수정 (원래 로직에 버그 있었음)
            int currentCell = src[currentIdx];
            if(currentCell == 1) {
                // 살아있는 셀: 2-3개 이웃(자기 제외)이면 생존
                dst[currentIdx] = (aliveCount == 3 || aliveCount == 4) ? 1 : 0;
            } else {
                // 죽은 셀: 정확히 3개 이웃이면 탄생
                dst[currentIdx] = (aliveCount == 3) ? 1 : 0;
            }
        }
    }
}

void GOL_2_H::padding(const std::vector<int>& pattern, std::vector<int>& board, int full_w, int full_h, int pattern_w, int pattern_h)
{
    // 보드 크기 설정 및 초기화
    board.resize(full_h * full_w);
    std::fill(board.begin(), board.end(), 0);

    int start_h = (full_h - pattern_h) / 2;
    int start_w = (full_w - pattern_w) / 2;

    // 패턴을 보드 중앙에 복사
    for(int h = 0; h < pattern_h; h++){
        for(int w = 0; w < pattern_w; w++){
            int patternIdx = h * pattern_w + w;
            int boardIdx = (h + start_h) * full_w + (w + start_w);
            board[boardIdx] = pattern[patternIdx];
        }
    }
}

int GOL_2_H::simulateAndLabel(const std::vector<int>& init_pattern)
{
    // 정적 변수로 메모리 재사용 (스레드별로 독립적)
    static thread_local std::vector<int> current;
    static thread_local std::vector<int> next;
    static thread_local std::vector<int> history;
    
    // 초기화
    current = init_pattern;
    next.resize(init_pattern.size());
    history.clear();
    
    const int window = 50;
    history.reserve(window + 10); // 메모리 미리 확보
    
    int constantCount = 0;
    int prev = -1;
    bool strictlyIncreasing = true;
    int gen = 0;

    while (gen < MAXGEN) {
        // 살아있는 셀 개수 계산 (최적화)
        int alive = 0;
        for(int cell : current) {
            alive += cell;
        }
        
        // 히스토리 관리 (메모리 효율적)
        if (static_cast<int>(history.size()) >= window) {
            history.erase(history.begin()); // deque의 pop_front() 대신
        }
        history.push_back(alive);
        
        // 종료 조건 검사
        if (prev == alive) {
            constantCount++;
        } else {
            constantCount = 0;
        }
        
        if (prev != -1 && alive <= prev) {
            strictlyIncreasing = false;
        }
        
        if (constantCount >= 100 || (strictlyIncreasing && gen >= 100)) {
            break;
        }
        
        prev = alive;
        
        // 다음 세대 계산 (in-place 방식으로 메모리 효율적)
        nextGen(current, next, BOARDWIDTH, BOARDHEIGHT);
        
        // 벡터 교체 (복사 없이 포인터만 교체)
        current.swap(next);
        
        gen++;
    }

    // 최종 살아있는 셀 개수
    int finalAlive = 0;
    for(int cell : current) {
        finalAlive += cell;
    }

    return finalAlive;
}

void GOL_2_H::generateGameOfLifeDataInHost(dataset_id data_config)
{
    // 프로젝트 루트 찾기 및 train_data 디렉토리 생성
    std::string projectRoot = findProjectRoot();
    std::string trainDataPath = projectRoot + "/train_data";
    std::filesystem::create_directories(trainDataPath);
    
    std::string datasetName = trainDataPath + "/" + getDatasetId(data_config);
    int totalFiles = data_config.sample_quantity;
    double aliveRatio = data_config.alive_ratio;

    std::cout << "=== CPU 버전 Game of Life 데이터 생성 ===" << std::endl;
    std::cout << "총 파일 수: " << totalFiles << " (파일명: " << datasetName << ")" << std::endl;
    std::cout << "생존 비율: " << aliveRatio << std::endl;
    std::cout << "최대 세대: " << MAXGEN << std::endl;
    std::cout << "패턴 크기: " << HEIGHT << " × " << WIDTH << std::endl;
    std::cout << "보드 크기: " << BOARDHEIGHT << " × " << BOARDWIDTH << std::endl;
    
    auto startTime = std::chrono::steady_clock::now();

    // 메모리 재사용을 위한 벡터들 (한번 할당 후 재사용)
    std::vector<int> pattern;
    std::vector<int> board;
    pattern.reserve(HEIGHT * WIDTH);
    board.reserve(BOARDHEIGHT * BOARDWIDTH);
    
    std::ofstream fout(datasetName + ".txt");
    if (!fout.is_open()) {
        std::cerr << "❌ 파일 생성 실패: " << datasetName << ".txt" << std::endl;
        std::cerr << "💡 디렉토리 권한을 확인하세요: " << trainDataPath << std::endl;
        return;
    }
    
    for (int fileId = 1; fileId <= totalFiles; ++fileId) {
        // 각 샘플에 고유하지만 결정적인 시드 생성
        uint64_t fileSeed = static_cast<uint64_t>(data_config.seed) ^ 
                           (static_cast<uint64_t>(fileId) * 0x9E3779B97F4A7C15ULL);
        
        // 패턴 생성 (벡터 재사용)
        genRandomPattern(pattern, static_cast<int>(fileSeed & 0x7fffffff), aliveRatio);
        
        // 패딩 적용 (벡터 재사용)
        padding(pattern, board, BOARDWIDTH, BOARDHEIGHT, WIDTH, HEIGHT);
        
        // 시뮬레이션 실행 및 레이블 생성
        int label = simulateAndLabel(board);
        
        // 파일 출력
        fout << "[" << fileId << "]" << '\n';
        
        // 초기 패턴 저장
        for (int i = 0; i < HEIGHT; ++i) {
            for (int j = 0; j < WIDTH; ++j) {
                fout << pattern[i * WIDTH + j];
            }
            fout << '\n';
        }
        fout << label << '\n';
        
        // 진행률 표시 최적화
        if (fileId % 100 == 0 || fileId == totalFiles) {
            std::string progressName = "CPU 데이터 생성 중... " + 
                                     std::to_string(fileId) + "/" + 
                                     std::to_string(totalFiles);
            printProgressBar(fileId, totalFiles, startTime, progressName);
        }
    }
    fout.close();
    
    std::cout << std::endl << "✅ [완료] 데이터셋 생성 완료" << std::endl;
    auto totalElapsed = std::chrono::steady_clock::now() - startTime;
    int totalSec = std::chrono::duration_cast<std::chrono::seconds>(totalElapsed).count();
    std::cout << "총 실행 시간: " << totalSec << " 초" << std::endl;
    
    // 메모리 사용량 정보 (참고용)
    size_t memoryUsed = (BOARDHEIGHT * BOARDWIDTH * 2 + HEIGHT * WIDTH) * sizeof(int);
    std::cout << "최적화된 메모리 사용량: " << memoryUsed / 1024 << " KB" << std::endl;
}
