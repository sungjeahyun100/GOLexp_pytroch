import numpy as np
import pygame #type: ignore
import torch
import torch.nn as nn
import torch.nn.functional as F
import ctypes as ct 
import argparse
import os
import glob
from datetime import datetime

# src 모듈에서 모델 임포트
from src.model import CNNLayer, get_activation, load_model, predict_to_number

# GPU 사용 가능 여부에 따라 .so 파일 선택
if torch.cuda.is_available():
    kernel_path = '../build/GOLdatagen_gpu.so'
    print("🚀 GPU 버전 라이브러리 로드: GOLdatagen_gpu.so")
else:
    kernel_path = '../build/GOLdatagen_cpu.so'
    print("💻 CPU 버전 라이브러리 로드: GOLdatagen_cpu.so")

lib = ct.CDLL(kernel_path)

lib.getPredict.argtypes = [ct.POINTER(ct.c_int)]
lib.getPredict.restype = ct.c_int

# 최적화된 함수 추가 (있다면)
try:
    lib.getPredictOptimized.argtypes = [ct.POINTER(ct.c_int)]
    lib.getPredictOptimized.restype = ct.c_int
    optimized_available = True
    print("🚀 최적화된 예측 함수 사용 가능")
except AttributeError:
    optimized_available = False
    print("⚠️ 최적화된 예측 함수를 찾을 수 없습니다")

def predict_actual(grid):
    flat = grid.flatten().astype(np.int32)
    ptr = flat.ctypes.data_as(ct.POINTER(ct.c_int))
    return lib.getPredict(ptr)

def predict_actual_optimized(grid):
    if not optimized_available:
        return predict_actual(grid)
    flat = grid.flatten().astype(np.int32)
    ptr = flat.ctypes.data_as(ct.POINTER(ct.c_int))
    return lib.getPredictOptimized(ptr)

def find_available_models(models_dir="saved_models"):
    """saved_models 폴더에서 사용 가능한 모델 파일들을 찾아서 반환"""
    if not os.path.exists(models_dir):
        print(f"❌ 모델 폴더를 찾을 수 없습니다: {models_dir}")
        return []
    
    model_files = glob.glob(os.path.join(models_dir, "*.pth"))
    model_files.sort(key=os.path.getmtime, reverse=True)  # 최신 파일 순으로 정렬
    
    if not model_files:
        print(f"❌ {models_dir} 폴더에 .pth 모델 파일이 없습니다.")
        return []
    
    print(f"📁 {len(model_files)}개의 모델 파일을 찾았습니다:")
    for i, model_path in enumerate(model_files):
        model_name = os.path.basename(model_path)
        mod_time = datetime.fromtimestamp(os.path.getmtime(model_path))
        print(f"  {i+1}. {model_name} (수정일: {mod_time.strftime('%Y-%m-%d %H:%M:%S')})")
    
    return model_files

def select_model_auto(model_files):
    """자동으로 최신 모델 선택 (GUI에서 선택 가능)"""
    if not model_files:
        return None
    
    # 최신 모델 자동 선택
    selected_model = model_files[0]
    print(f"🚀 최신 모델 자동 선택: {os.path.basename(selected_model)}")
    print(f"💡 인터페이스에서 M키를 눌러 다른 모델로 변경할 수 있습니다.")
    return selected_model

def get_model_path_smart(provided_path=None):
    """모델 경로를 스마트하게 결정하는 함수"""
    # 1. 명시적으로 경로가 제공된 경우
    if provided_path:
        if os.path.exists(provided_path):
            print(f"✅ 지정된 모델 사용: {provided_path}")
            return provided_path
        else:
            print(f"❌ 지정된 모델을 찾을 수 없습니다: {provided_path}")
            print("📁 saved_models 폴더에서 대체 모델을 찾습니다...")
    
    # 2. saved_models 폴더에서 자동 탐색
    model_files = find_available_models()
    
    if not model_files:
        print("❌ 사용 가능한 모델이 없습니다.")
        return None
    
    # 3. 자동 모델 선택 (GUI에서 변경 가능)
    return select_model_auto(model_files)

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
    def __init__(self, model_path=None):
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
        
        # 모델 관리 초기화
        self.model = None
        self.model_info = {}
        self.current_model_path = model_path
        self.available_models = find_available_models()
        self.selected_model_index = 0
        
        # 모델 로드 (경로가 제공된 경우에만)
        if model_path:
            self.load_model(model_path)
            # 모델 리스트에서 현재 선택된 인덱스 찾기
            for i, available_path in enumerate(self.available_models):
                if available_path == model_path:
                    self.selected_model_index = i
                    break
        
        # 예측 결과
        self.predictions = np.zeros(10, dtype=np.float32)
        self.predicted_decimal = 0
        
        # UI 상태
        self.mode = "predict"
        self.show_model_selector = False
        self.collected_patterns = []
    
    def load_model(self, model_path):
        """훈련된 모델 로드 - 스마트 호환성 지원"""
        model_name = os.path.basename(model_path)
        try:
            if torch.cuda.is_available():
                checkpoint = torch.load(model_path)
            else:
                checkpoint = torch.load(model_path, map_location='cpu')
            
            self.model_info = checkpoint.get('model_info', {})
            
            # 모델 구조 파라미터 동적 추출 (호환성 지원)
            input_size = self.model_info.get('input_size', 50)
            hidden1_size = self.model_info.get('hidden1_size', self.model_info.get('hidden1', 32))
            hidden2_size = self.model_info.get('hidden2_size', self.model_info.get('hidden2', 64))
            output_size = self.model_info.get('output_size', 10)
            
            # activation vs activate 키 호환성
            activation = (self.model_info.get('activation') or 
                         self.model_info.get('activate') or 'swish')
            
            stride = self.model_info.get('stride', 1)
            use_bias = self.model_info.get('use_bias', False)
            
            print(f"🔄 모델 로딩 시도: {model_name}")
            print(f"   구조: {input_size}x{input_size} → {hidden1_size}→{hidden2_size} → {output_size}")
            print(f"   활성화: {activation}, bias: {use_bias}")
            
            # 여러 가지 모델 구조로 시도
            success = False
            
            # 1. 먼저 현재 구조 (풀링 없음)로 시도
            if not success:
                try:
                    self.model = CNNLayer(
                        input_size=input_size,
                        hidden1_size=hidden1_size,
                        hidden2_size=hidden2_size,
                        output_size=output_size,
                        activate=activation,
                        stride=stride,
                        use_bias=use_bias
                    ).to(device)
                    
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"✅ 풀링 없는 모델로 로드 성공")
                    success = True
                except Exception as e:
                    print(f"⚠️ 풀링 없는 구조 실패: {str(e)[:50]}...")
            
            # 2. 실패하면 다른 input_size로 시도 (50 ↔ 10)
            if not success:
                try:
                    alt_input_size = 50 if input_size == 10 else 10
                    self.model = CNNLayer(
                        input_size=alt_input_size,
                        hidden1_size=hidden1_size,
                        hidden2_size=hidden2_size,
                        output_size=output_size,
                        activate=activation,
                        stride=stride,
                        use_bias=use_bias
                    ).to(device)
                    
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"✅ 대체 입력크기({alt_input_size})로 로드 성공")
                    success = True
                except Exception as e:
                    print(f"⚠️ 대체 입력크기 실패: {str(e)[:50]}...")
            
            if success and self.model is not None:
                self.model.eval()
                print(f"🎯 모델명: {self.model_info.get('name', '알 수 없음')}")
                if 'description' in self.model_info:
                    print(f"📝 설명: {self.model_info['description']}")
            else:
                print(f"❌ 모든 로딩 시도 실패")
                self.model = None
                
        except Exception as e:
            print(f"❌ 모델 파일 로드 오류: {model_name}")
            print(f"   오류: {str(e)[:100]}")
            print("   더미 예측을 사용합니다.")
            self.model = None
            self.model_info = {}
    
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
                    self.show_model_selector = False
                    print("예측 모드로 변경")
                elif event.key == pygame.K_t:
                    self.mode = "train"
                    self.show_model_selector = False
                    print("훈련 모드로 변경")
                elif event.key == pygame.K_d:
                    self.mode = "collect"
                    self.show_model_selector = False
                    print("데이터 수집 모드로 변경")
                elif event.key == pygame.K_m:
                    self.show_model_selector = not self.show_model_selector
                    print("모델 선택기 토글")
                elif event.key == pygame.K_UP and self.show_model_selector:
                    if self.available_models:
                        self.selected_model_index = (self.selected_model_index - 1) % len(self.available_models)
                elif event.key == pygame.K_DOWN and self.show_model_selector:
                    if self.available_models:
                        self.selected_model_index = (self.selected_model_index + 1) % len(self.available_models)
                elif event.key == pygame.K_RETURN and self.show_model_selector:
                    if self.available_models and 0 <= self.selected_model_index < len(self.available_models):
                        selected_model = self.available_models[self.selected_model_index]
                        self.current_model_path = selected_model
                        self.load_model(selected_model)
                        self.show_model_selector = False
                        print(f"모델 변경: {os.path.basename(selected_model)}")
                elif event.key == pygame.K_ESCAPE:
                    self.show_model_selector = False
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
        
        # AI 모델 상태 및 이름
        if self.model and self.current_model_path:
            model_name = os.path.basename(self.current_model_path)
            # 긴 모델 이름을 줄여서 표시
            if len(model_name) > 25:
                model_name = model_name[:22] + "..."
            model_status = f"Model: {model_name}"
            model_color = GREEN
        else:
            model_status = "AI Model: NOT LOADED"
            model_color = RED
            
        text = self.small_font.render(model_status, True, model_color)
        self.screen.blit(text, (WINDOW_SIZE + 10, y_offset))
        y_offset += 20
        
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
            "M: Model selector", "A: Add pattern", "X: Clear patterns", 
            "C: Clear grid", "R: Random grid"
        ]
        
        for shortcut in shortcuts:
            color = WHITE if shortcut == "Shortcuts:" else GRAY
            font_to_use = self.small_font if shortcut != "Shortcuts:" else self.font
            text = font_to_use.render(shortcut, True, color)
            self.screen.blit(text, (WINDOW_SIZE + 10, y_offset))
            y_offset += 16 if shortcut != "Shortcuts:" else 20
        
        y_offset += 10
        
        # 모델 선택기 UI
        if self.show_model_selector:
            self.draw_model_selector(y_offset)
        # 모드별 UI
        elif self.mode == "predict":
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
    
    def draw_model_selector(self, y_start):
        """모델 선택기 UI 그리기"""
        # 배경
        selector_rect = pygame.Rect(WINDOW_SIZE + 5, y_start - 5, SIDEBAR_WIDTH - 10, WINDOW_SIZE - y_start - 10)
        pygame.draw.rect(self.screen, (30, 30, 30), selector_rect)
        pygame.draw.rect(self.screen, WHITE, selector_rect, 2)
        
        # 제목
        title_text = self.large_font.render("Model Selector", True, YELLOW)
        self.screen.blit(title_text, (WINDOW_SIZE + 15, y_start + 5))
        y_offset = y_start + 35
        
        # 안내 텍스트
        guide_texts = [
            "↑↓: Navigate",
            "Enter: Select", 
            "Esc: Cancel"
        ]
        for guide in guide_texts:
            text = self.small_font.render(guide, True, GRAY)
            self.screen.blit(text, (WINDOW_SIZE + 15, y_offset))
            y_offset += 14
        
        y_offset += 10
        
        # 모델 리스트
        if not self.available_models:
            no_models_text = self.font.render("No models found", True, RED)
            self.screen.blit(no_models_text, (WINDOW_SIZE + 15, y_offset))
            return
        
        # 스크롤 가능한 모델 리스트 (최대 8개 표시)
        max_visible = 8
        start_idx = max(0, self.selected_model_index - max_visible // 2)
        end_idx = min(len(self.available_models), start_idx + max_visible)
        
        for i in range(start_idx, end_idx):
            model_path = self.available_models[i]
            model_name = os.path.basename(model_path)
            
            # 긴 이름 줄이기
            if len(model_name) > 28:
                model_name = model_name[:25] + "..."
            
            # 선택된 모델 하이라이트
            if i == self.selected_model_index:
                # 배경 하이라이트
                highlight_rect = pygame.Rect(WINDOW_SIZE + 10, y_offset - 2, SIDEBAR_WIDTH - 20, 18)
                pygame.draw.rect(self.screen, BLUE, highlight_rect)
                text_color = WHITE
                prefix = "► "
            else:
                text_color = WHITE if model_path == self.current_model_path else GRAY
                prefix = "  "
            
            # 모델 이름 출력
            display_text = f"{prefix}{model_name}"
            text = self.small_font.render(display_text, True, text_color)
            self.screen.blit(text, (WINDOW_SIZE + 15, y_offset))
            
            y_offset += 18
        
        # 스크롤 인디케이터
        if len(self.available_models) > max_visible:
            scroll_text = f"({self.selected_model_index + 1}/{len(self.available_models)})"
            text = self.small_font.render(scroll_text, True, GRAY)
            self.screen.blit(text, (WINDOW_SIZE + 15, y_offset + 5))
    
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
    parser.add_argument('--model_name', '-m', type=str, help='모델 이름 (선택사항)')

    args = parser.parse_args()
    
    # 스마트 모델 경로 결정
    model_path = get_model_path_smart(args.model_name)
    
    interface = GameOfLifeInterface(model_path)
    interface.run()