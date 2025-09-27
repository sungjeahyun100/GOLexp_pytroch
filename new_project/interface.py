import numpy as np
import pygame #type: ignore
import torch
import torch.nn as nn
import torch.nn.functional as F
import ctypes as ct 
import argparse

# src 모듈에서 모델 임포트
from src.model import CNNLayer, get_activation, load_model, predict_to_number

kernel_path = '../build/GOLdatagen.so'

lib = ct.CDLL(kernel_path)

lib.getPredict.argtypes = [ct.POINTER(ct.c_int)]
lib.getPredict.restype = ct.c_int

def predict_actual(grid):
    flat = grid.flatten().astype(np.int32)
    ptr = flat.ctypes.data_as(ct.POINTER(ct.c_int))
    return lib.getPredict(ptr)

# 기존 색상 및 설정 유지...
WINDOW_SIZE = 600
GRID_SIZE = 10
CELL_SIZE = WINDOW_SIZE // GRID_SIZE
PADDING = 20
SIDEBAR_WIDTH = 300

# 색상 정의 (기존과 동일)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
GREEN = (0, 255, 100)
DARK_GREEN = (0, 180, 70)
DARK_GRAY = (40, 40, 40)
BLUE = (50, 150, 255)
YELLOW = (255, 255, 0)
RED = (255, 100, 100)
PURPLE = (200, 100, 255)

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 활성화 함수 (모델에서 가져옴)
def getActive(activeName, input):
    if activeName == 'tanh':
        return F.tanh(input)
    elif activeName == 'relu':
        return F.relu(input)
    elif activeName == 'sigmoid':
        return F.sigmoid(input)
    elif activeName == 'elu':
        return F.elu(input)
    elif activeName == 'selu':
        return F.selu(input)
    elif activeName == 'lrelu':
        return F.leaky_relu(input)
    elif activeName == 'swish':
        return F.silu(input)
    else:
        return F.tanh(input)


# 모델 정의는 이제 src.model에서 임포트됨

class GameOfLifeInterface:
    def __init__(self, model_path="saved_models/cnn_gol_model1.pth"):
        # 파이게임 초기화
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_SIZE + SIDEBAR_WIDTH, WINDOW_SIZE))
        pygame.display.set_caption("Game of Life AI Interface")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 16)
        self.large_font = pygame.font.SysFont('Arial', 24, bold=True)
        self.small_font = pygame.font.SysFont('Arial', 12)
        
        # 그리드 초기화
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        
        # AI 모델 로드
        self.model = None
        self.model_info = {}
        self.load_model(model_path)
        
        # 예측 결과
        self.predictions = np.zeros(10, dtype=np.float32)
        self.predicted_decimal = 0
        
        # UI 상태
        self.mode = "predict"
        self.collected_patterns = []
    
    def load_model(self, model_path):
        """훈련된 모델 로드"""
        try:
            if torch.cuda.is_available():
                checkpoint = torch.load(model_path)
            else:
                checkpoint = torch.load(model_path, map_location='cpu')
            
            self.model_info = checkpoint.get('model_info', {})
            
            # 모델 생성
            self.model = CNNLayer(
                input_size=self.model_info.get('input_size', 50),
                hidden1_size=self.model_info.get('hidden1_size', 32),
                hidden2_size=self.model_info.get('hidden2_size', 64),
                output_size=self.model_info.get('output_size', 10),
                activate=self.model_info.get('activate', 'swish'),
                stride=self.model_info.get('stride', 1),
                use_bias=False
            ).to(device)
            
            # 가중치 로드
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            print(f"모델 로드 성공: {model_path}")
            print(f"모델 정보: {self.model_info}")
            
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            print("더미 예측을 사용합니다.")
            self.model = None
    
    def predict_with_model(self):
        """AI 모델로 예측"""
        if self.model is None:
            # 더미 예측
            grid_sum = np.sum(self.grid)
            for i in range(8):
                self.predictions[i] = np.random.random() if grid_sum > 0 else 0
            return
        
        try:
            # 그리드를 텐서로 변환
            grid_tensor = torch.tensor(self.grid, dtype=torch.float32).unsqueeze(0).to(device)
            
            # 모델 예측
            with torch.no_grad():
                output = self.model(grid_tensor)
                self.predictions = output.cpu().numpy().flatten()
            
            # 10bit → 10진수 변환
            binary_output = (self.predictions > 0.5).astype(int)
            self.predicted_decimal = sum(bit * (2 ** (9-i)) for i, bit in enumerate(binary_output))
            
        except Exception as e:
            print(f"예측 오류: {e}")
            # 오류 시 더미 예측
            self.predictions = np.zeros(10, dtype=np.float32)
            self.predicted_decimal = 0
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    self.mode = "predict"
                    print("예측 모드로 변경")
                elif event.key == pygame.K_t:
                    self.mode = "train"
                    print("훈련 모드로 변경")
                elif event.key == pygame.K_d:
                    self.mode = "collect"
                    print("데이터 수집 모드로 변경")
                elif event.key == pygame.K_a:
                    if self.mode == "collect":
                        self.add_pattern()
                elif event.key == pygame.K_x:
                    self.clear_patterns()
                elif event.key == pygame.K_c:
                    self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
                    if self.mode == "predict":
                        self.predict_with_model()
                elif event.key == pygame.K_r:
                    self.grid = np.random.choice([0, 1], size=(GRID_SIZE, GRID_SIZE)).astype(np.float32)
                    if self.mode == "predict":
                        self.predict_with_model()
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                
                if x < WINDOW_SIZE and y < WINDOW_SIZE:
                    grid_x = x // CELL_SIZE
                    grid_y = y // CELL_SIZE
                    
                    self.grid[grid_y, grid_x] = 1 - self.grid[grid_y, grid_x]
                    
                    if self.mode == "predict":
                        self.predict_with_model()
        
        return True
    
    def add_pattern(self):
        if np.sum(self.grid) == 0:
            print("빈 그리드는 추가할 수 없습니다.")
            return
        
        pattern = self.grid.copy()
        self.collected_patterns.append(pattern)
        print(f"패턴 추가됨 (총 {len(self.collected_patterns)}개)")
    
    def clear_patterns(self):
        self.collected_patterns.clear()
        print("수집된 패턴들이 삭제되었습니다.")
    
    def draw_grid(self):
        self.screen.fill(BLACK)
        
        # 그리드 그리기
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                
                if self.grid[y, x] > 0:
                    pygame.draw.rect(self.screen, GREEN, rect)
                else:
                    pygame.draw.rect(self.screen, DARK_GRAY, rect)
                
                pygame.draw.rect(self.screen, GRAY, rect, 1)
        
        # 사이드바
        sidebar_rect = pygame.Rect(WINDOW_SIZE, 0, SIDEBAR_WIDTH, WINDOW_SIZE)
        pygame.draw.rect(self.screen, DARK_GRAY, sidebar_rect)
        
        self.draw_ui()
    
    def draw_ui(self):
        y_offset = 20
        
        # AI 모델 상태
        model_status = "AI Model: LOADED" if self.model else "AI Model: NOT LOADED"
        model_color = GREEN if self.model else RED
        text = self.font.render(model_status, True, model_color)
        self.screen.blit(text, (WINDOW_SIZE + 10, y_offset))
        y_offset += 25
        
        # 현재 모드
        mode_text = f"Mode: {self.mode.capitalize()}"
        mode_color = GREEN if self.mode == "predict" else (YELLOW if self.mode == "train" else PURPLE)
        text = self.font.render(mode_text, True, mode_color)
        self.screen.blit(text, (WINDOW_SIZE + 10, y_offset))
        y_offset += 30
        
        # 단축키
        shortcuts = [
            "Shortcuts:",
            "P: Predict mode", "T: Train mode", "D: Data collection mode",
            "A: Add pattern", "X: Clear patterns", "C: Clear grid", "R: Random grid"
        ]
        
        for shortcut in shortcuts:
            color = WHITE if shortcut == "Shortcuts:" else GRAY
            font_to_use = self.small_font if shortcut != "Shortcuts:" else self.font
            text = font_to_use.render(shortcut, True, color)
            self.screen.blit(text, (WINDOW_SIZE + 10, y_offset))
            y_offset += 16 if shortcut != "Shortcuts:" else 20
        
        y_offset += 10
        
        # 모드별 UI
        if self.mode == "predict":
            self.draw_prediction_results(y_offset)
        elif self.mode == "collect":
            pattern_count_text = f"Collected: {len(self.collected_patterns)} patterns"
            text = self.font.render(pattern_count_text, True, PURPLE)
            self.screen.blit(text, (WINDOW_SIZE + 10, y_offset))
    
    def draw_prediction_results(self, y_start):
        if not self.model:
            no_model_text = self.font.render("No AI model loaded", True, RED)
            self.screen.blit(no_model_text, (WINDOW_SIZE + 10, y_start))
            return
        
        title = self.font.render("AI Prediction:", True, WHITE)
        self.screen.blit(title, (WINDOW_SIZE + 10, y_start))
        y_start += 25
        
        # 10bit 값들
        for i, pred in enumerate(self.predictions):
            bit_value = 1 if pred > 0.5 else 0
            value_text = f"Bit {i}: {pred:.3f} -> {bit_value}"
            color = GREEN if pred > 0.5 else WHITE
            text = self.small_font.render(value_text, True, color)
            self.screen.blit(text, (WINDOW_SIZE + 10, y_start + i*16))
        
        y_start += 10 * 16 + 10
        
        # 이진수 표시
        binary_str = ''.join(['1' if p > 0.5 else '0' for p in self.predictions])
        binary_text = self.font.render(f"Binary: {binary_str}", True, YELLOW)
        self.screen.blit(binary_text, (WINDOW_SIZE + 10, y_start))
        
        # 10진수 표시
        decimal_text = self.large_font.render(f"Result: {self.predicted_decimal}", True, YELLOW)
        self.screen.blit(decimal_text, (WINDOW_SIZE + 10, y_start + 25))
        
        # 모델 정보
        if self.model_info:
            y_start += 60
            info_text = self.small_font.render("Model Info:", True, GRAY)
            self.screen.blit(info_text, (WINDOW_SIZE + 10, y_start))
            
            samples_text = f"Training samples: {self.model_info.get('training_samples', 'Unknown')}"
            epochs_text = f"Training epochs: {self.model_info.get('training_epochs', 'Unknown')}"
            
            info_lines = [samples_text, epochs_text]
            for i, line in enumerate(info_lines):
                text = self.small_font.render(line, True, GRAY)
                self.screen.blit(text, (WINDOW_SIZE + 10, y_start + 15 + i*14))
    
    def run(self):
        running = True
        print("Game of Life AI Interface 시작")
        print("모델이 로드되면 P키로 예측 모드에서 테스트하세요!")
        
        while running:
            running = self.handle_events()
            self.draw_grid()
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()

# 실행
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Game of Life interface')

    parser.add_argument('--model_name', '-m', type=str, help='모델 이름')

    args = parser.parse_args()
    # 저장된 모델 경로
    model_path = args.model_name
    
    interface = GameOfLifeInterface(model_path)
    interface.run()