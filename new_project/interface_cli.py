#!/usr/bin/env python3
"""
Game of Life CLI Interface - 윈도우 사용자를 위한 텍스트 기반 인터페이스

PyGame 없이도 모델 테스트와 패턴 실험이 가능한 CLI 버전입니다.

사용법:
    python3 interface_cli.py
    python3 interface_cli.py --model saved_models/my_model.pth
    python3 interface_cli.py --headless  # 자동 테스트 모드
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ctypes as ct 
import argparse
import os
import glob
import json
import re
from datetime import datetime
import sys

# src 모듈에서 모델 임포트
from src.model import CNNLayer, get_activation, load_model, predict_to_number

# GPU 사용 가능 여부에 따라 .so 파일 선택
if torch.cuda.is_available():
    kernel_path = '../build/GOLdatagen_gpu.so'
    print("🚀 GPU 버전 라이브러리 로드: GOLdatagen_gpu.so")
else:
    kernel_path = '../build/GOLdatagen_cpu.so'
    print("💻 CPU 버전 라이브러리 로드: GOLdatagen_cpu.so")

# C++ 라이브러리 로드
try:
    lib = ct.CDLL(kernel_path)
    lib.getPredict.argtypes = [ct.POINTER(ct.c_int)]
    lib.getPredict.restype = ct.c_int
    
    # 최적화된 함수도 시도
    try:
        lib.getPredictOptimized.argtypes = [ct.POINTER(ct.c_int)]
        lib.getPredictOptimized.restype = ct.c_int
        optimized_available = True
        print("🚀 최적화된 예측 함수 사용 가능")
    except AttributeError:
        optimized_available = False
        print("⚠️ 최적화된 예측 함수를 찾을 수 없습니다")
    
    library_loaded = True
except Exception as e:
    print(f"⚠️ C++ 라이브러리 로드 실패: {e}")
    print("   시뮬레이션 기능이 제한됩니다.")
    library_loaded = False
    optimized_available = False

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict_actual(grid):
    """실제 Game of Life 시뮬레이션 실행"""
    if not library_loaded:
        return -1
    
    flat = grid.flatten().astype(np.int32)
    ptr = flat.ctypes.data_as(ct.POINTER(ct.c_int))
    return lib.getPredict(ptr)

def predict_actual_optimized(grid):
    """최적화된 시뮬레이션 실행"""
    if not optimized_available:
        return predict_actual(grid)
    
    flat = grid.flatten().astype(np.int32)
    ptr = flat.ctypes.data_as(ct.POINTER(ct.c_int))
    return lib.getPredictOptimized(ptr)

class CLIGameOfLife:
    """CLI 기반 Game of Life 인터페이스"""
    
    def __init__(self, model_path=None, headless=False):
        self.grid = np.zeros((10, 10), dtype=np.float32)
        self.model = None
        self.model_info = {}
        self.current_model_path = model_path
        self.headless = headless
        
        # 사용 가능한 모델들 찾기
        self.available_models = self.find_available_models()
        
        # 모델 로드
        if model_path:
            self.load_model(model_path)
        elif self.available_models:
            # 최신 모델 자동 선택
            latest_model = self.available_models[0]
            print(f"🎯 최신 모델 자동 선택: {os.path.basename(latest_model)}")
            self.load_model(latest_model)
        else:
            print("❌ 사용 가능한 모델이 없습니다.")
        
        # 저장된 패턴들
        self.saved_patterns = {}
        self.load_saved_patterns()
        
        # 미리 정의된 패턴 라이브러리
        self.pattern_library = {}
        self.load_pattern_library()
    
    def find_available_models(self, models_dir="saved_models"):
        """사용 가능한 모델 파일들 찾기"""
        if not os.path.exists(models_dir):
            return []
        
        model_files = glob.glob(os.path.join(models_dir, "*.pth"))
        model_files.sort(key=os.path.getmtime, reverse=True)  # 최신 순
        return model_files
    
    def load_model(self, model_path):
        """모델 로드"""
        model_name = os.path.basename(model_path)
        try:
            if torch.cuda.is_available():
                checkpoint = torch.load(model_path)
            else:
                checkpoint = torch.load(model_path, map_location='cpu')
            
            self.model_info = checkpoint.get('model_info', {})
            
            # 모델 구조 파라미터 추출
            input_size = self.model_info.get('input_size', 10)
            hidden1_size = self.model_info.get('hidden1_size', self.model_info.get('hidden1', 32))
            hidden2_size = self.model_info.get('hidden2_size', self.model_info.get('hidden2', 64))
            output_size = self.model_info.get('output_size', 10)
            activation = self.model_info.get('activation', self.model_info.get('activate', 'swish'))
            stride = self.model_info.get('stride', 1)
            use_bias = self.model_info.get('use_bias', False)
            
            print(f"🔄 모델 로딩: {model_name}")
            print(f"   구조: {input_size}x{input_size} → {hidden1_size}→{hidden2_size} → {output_size}")
            
            # 모델 생성 및 로드
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
            self.model.eval()
            self.current_model_path = model_path
            
            print(f"✅ 모델 로드 성공!")
            if 'epochs' in self.model_info:
                print(f"   훈련 에포크: {self.model_info['epochs']}")
            if 'learning_rate' in self.model_info:
                print(f"   학습률: {self.model_info['learning_rate']}")
                
        except Exception as e:
            print(f"❌ 모델 로드 실패: {model_name}")
            print(f"   오류: {str(e)}")
            self.model = None
            self.model_info = {}
    
    def predict_with_model(self):
        """AI 모델로 예측"""
        if self.model is None:
            return None, "모델이 로드되지 않았습니다"
        
        try:
            # 그리드를 텐서로 변환
            grid_tensor = torch.tensor(self.grid, dtype=torch.float32).unsqueeze(0).to(device)
            
            # 모델 예측
            with torch.no_grad():
                output = self.model(grid_tensor)
                predictions = output.cpu().numpy().flatten()
            
            # 10bit → 10진수 변환
            binary_output = (predictions > 0.5).astype(int)
            decimal_result = sum(bit * (2 ** (9-i)) for i, bit in enumerate(binary_output))
            
            return {
                'predictions': predictions,
                'binary': binary_output,
                'decimal': decimal_result,
                'binary_str': ''.join(map(str, binary_output))
            }, None
            
        except Exception as e:
            return None, f"예측 오류: {str(e)}"
    
    def print_grid(self, title="Current Grid"):
        """그리드를 텍스트로 출력"""
        print(f"\n=== {title} ===")
        print("   " + "".join([f"{i:2}" for i in range(10)]))
        for i, row in enumerate(self.grid):
            row_str = f"{i:2} "
            for cell in row:
                if cell > 0:
                    row_str += "██"  # 살아있는 셀
                else:
                    row_str += "  "  # 죽은 셀
            print(row_str)
        print()
    
    def edit_grid_interactive(self):
        """대화형 그리드 편집"""
        print("\n🎮 그리드 편집 모드")
        print("좌표를 입력하여 셀 상태를 변경하세요 (예: 3,4)")
        print("명령어: 'show' (그리드 보기), 'clear' (모두 지우기), 'random' (랜덤), 'done' (완료)")
        
        while True:
            self.print_grid()
            command = input("입력 (좌표 또는 명령어) > ").strip().lower()
            
            if command == 'done':
                break
            elif command == 'show':
                continue
            elif command == 'clear':
                self.grid = np.zeros((10, 10), dtype=np.float32)
                print("그리드가 초기화되었습니다.")
            elif command == 'random':
                density = input("셀 밀도 (0.0-1.0, 기본값 0.3): ").strip()
                try:
                    density = float(density) if density else 0.3
                    density = max(0.0, min(1.0, density))
                    self.grid = np.random.choice([0, 1], size=(10, 10), p=[1-density, density]).astype(np.float32)
                    print(f"랜덤 그리드 생성 (밀도: {density})")
                except ValueError:
                    print("잘못된 밀도 값입니다.")
            else:
                # 좌표 입력 처리
                try:
                    if ',' in command:
                        x, y = map(int, command.split(','))
                        if 0 <= x < 10 and 0 <= y < 10:
                            self.grid[y, x] = 1 - self.grid[y, x]
                            state = "활성화" if self.grid[y, x] > 0 else "비활성화"
                            print(f"셀 ({x},{y}) {state}")
                        else:
                            print("좌표는 0-9 범위여야 합니다.")
                    else:
                        print("좌표 형식: x,y (예: 3,4)")
                except ValueError:
                    print("잘못된 좌표 형식입니다. 예: 3,4")
    
    def load_saved_patterns(self):
        """저장된 패턴들 로드"""
        pattern_file = "cli_patterns.json"
        if os.path.exists(pattern_file):
            try:
                with open(pattern_file, 'r', encoding='utf-8') as f:
                    self.saved_patterns = json.load(f)
                print(f"📁 {len(self.saved_patterns)}개 저장된 패턴을 로드했습니다.")
            except Exception as e:
                print(f"⚠️ 패턴 로드 실패: {e}")
                self.saved_patterns = {}
    
    def load_pattern_library(self):
        """미리 정의된 패턴 라이브러리 로드"""
        library_file = "gol_patterns.json"
        if os.path.exists(library_file):
            try:
                with open(library_file, 'r', encoding='utf-8') as f:
                    self.pattern_library = json.load(f)
                
                # 패턴 개수 계산
                total_patterns = 0
                for category in self.pattern_library.values():
                    if isinstance(category, dict) and not category.get('_description'):
                        total_patterns += len([k for k in category.keys() if not k.startswith('_')])
                
                print(f"📚 {total_patterns}개 라이브러리 패턴을 로드했습니다.")
            except Exception as e:
                print(f"⚠️ 패턴 라이브러리 로드 실패: {e}")
                self.pattern_library = {}
    
    def save_patterns(self):
        """패턴들을 파일에 저장"""
        pattern_file = "cli_patterns.json"
        try:
            with open(pattern_file, 'w', encoding='utf-8') as f:
                json.dump(self.saved_patterns, f, indent=2, ensure_ascii=False)
            print(f"💾 패턴이 {pattern_file}에 저장되었습니다.")
        except Exception as e:
            print(f"❌ 패턴 저장 실패: {e}")
    
    def save_current_pattern(self):
        """현재 그리드를 패턴으로 저장"""
        if np.sum(self.grid) == 0:
            print("❌ 빈 그리드는 저장할 수 없습니다.")
            return
        
        name = input("패턴 이름을 입력하세요: ").strip()
        if not name:
            print("❌ 패턴 이름이 필요합니다.")
            return
        
        self.saved_patterns[name] = {
            'grid': self.grid.tolist(),
            'created_at': datetime.now().isoformat(),
            'description': input("설명 (선택사항): ").strip()
        }
        
        self.save_patterns()
        print(f"✅ 패턴 '{name}'이 저장되었습니다.")
    
    def load_pattern(self, pattern_name):
        """저장된 패턴 로드"""
        if pattern_name not in self.saved_patterns:
            print(f"❌ 패턴 '{pattern_name}'을 찾을 수 없습니다.")
            return False
        
        pattern_data = self.saved_patterns[pattern_name]
        self.grid = np.array(pattern_data['grid'], dtype=np.float32)
        print(f"✅ 패턴 '{pattern_name}'을 로드했습니다.")
        if pattern_data.get('description'):
            print(f"   설명: {pattern_data['description']}")
        return True
    
    def list_saved_patterns(self):
        """저장된 패턴 목록 출력"""
        if not self.saved_patterns:
            print("저장된 패턴이 없습니다.")
            return
        
        print(f"\n📚 저장된 패턴 ({len(self.saved_patterns)}개):")
        for i, (name, data) in enumerate(self.saved_patterns.items(), 1):
            created = data.get('created_at', 'Unknown')[:10]  # 날짜만
            desc = data.get('description', '')[:50]  # 설명 50자만
            print(f"  {i:2d}. {name} ({created})")
            if desc:
                print(f"      {desc}")
    
    def list_pattern_library(self):
        """패턴 라이브러리 목록 출력"""
        if not self.pattern_library:
            print("패턴 라이브러리가 없습니다.")
            return
        
        print(f"\n📚 GoL 패턴 라이브러리:")
        
        for category_name, category_data in self.pattern_library.items():
            if category_name.startswith('_'):
                continue
                
            if not isinstance(category_data, dict):
                continue
            
            # 카테고리 제목과 설명
            category_desc = category_data.get('_description', '')
            patterns = {k: v for k, v in category_data.items() if not k.startswith('_')}
            
            if patterns:
                print(f"\n🏷️ {category_name.upper()} ({len(patterns)}개)")
                if category_desc:
                    print(f"   {category_desc}")
                
                for i, (pattern_name, pattern_data) in enumerate(patterns.items(), 1):
                    name = pattern_data.get('name', pattern_name)
                    desc = pattern_data.get('description', '')[:60]
                    period = pattern_data.get('period', '?')
                    type_info = pattern_data.get('type', '')
                    
                    period_str = f"주기-{period}" if period > 0 else "불안정" if period == 0 else ""
                    
                    print(f"  {i:2d}. {pattern_name} - {name}")
                    print(f"      {desc} ({period_str})")
    
    def load_library_pattern(self, category, pattern_name):
        """라이브러리에서 패턴 로드"""
        if category not in self.pattern_library:
            print(f"❌ 카테고리 '{category}'를 찾을 수 없습니다.")
            return False
        
        category_data = self.pattern_library[category]
        if pattern_name not in category_data:
            print(f"❌ 패턴 '{pattern_name}'을 '{category}' 카테고리에서 찾을 수 없습니다.")
            return False
        
        pattern_data = category_data[pattern_name]
        pattern = pattern_data['pattern']
        
        # 패턴을 10x10 그리드 중앙에 배치
        self.grid = np.zeros((10, 10), dtype=np.float32)
        
        pattern_array = np.array(pattern, dtype=np.float32)
        pattern_height, pattern_width = pattern_array.shape
        
        # 중앙 배치 계산
        start_y = max(0, (10 - pattern_height) // 2)
        start_x = max(0, (10 - pattern_width) // 2)
        
        end_y = min(10, start_y + pattern_height)
        end_x = min(10, start_x + pattern_width)
        
        # 패턴이 10x10을 초과하는 경우 잘라내기
        crop_height = end_y - start_y
        crop_width = end_x - start_x
        
        self.grid[start_y:end_y, start_x:end_x] = pattern_array[:crop_height, :crop_width]
        
        name = pattern_data.get('name', pattern_name)
        desc = pattern_data.get('description', '')
        period = pattern_data.get('period', '?')
        
        print(f"✅ 패턴 '{name}' 로드 완료")
        print(f"   설명: {desc}")
        if period != '?':
            period_str = f"주기-{period}" if period > 0 else "불안정" if period == 0 else "안정"
            print(f"   특성: {period_str}")
        
        if pattern_height > 10 or pattern_width > 10:
            print(f"⚠️ 패턴이 잘렸습니다. 원본 크기: {pattern_height}x{pattern_width}")
        
        return True
    
    def run_prediction_test(self):
        """예측 테스트 실행"""
        print("\n🎯 예측 테스트 모드")
        
        if np.sum(self.grid) == 0:
            print("❌ 그리드가 비어있습니다. 먼저 패턴을 만들어주세요.")
            return
        
        self.print_grid("테스트할 패턴")
        
        # AI 모델 예측
        ai_result, error = self.predict_with_model()
        if error or ai_result is None:
            print(f"❌ AI 예측 실패: {error}")
            return
        
        print(f"\n🤖 AI 모델 예측:")
        print(f"   이진수: {ai_result['binary_str']}")
        print(f"   십진수: {ai_result['decimal']}")
        
        # 각 비트별 신뢰도 출력
        print("\n   비트별 신뢰도:")
        for i, (prob, bit) in enumerate(zip(ai_result['predictions'], ai_result['binary'])):
            confidence = prob if bit == 1 else (1 - prob)
            print(f"   Bit {i}: {bit} (신뢰도: {confidence:.3f})")
        
        # 실제 시뮬레이션과 비교
        if library_loaded:
            print("\n⚙️ 실제 시뮬레이션 실행중...")
            actual_result = predict_actual(self.grid)
            
            if actual_result >= 0:
                print(f"🎲 실제 결과: {actual_result}")
                
                # 비교 결과
                diff = abs(ai_result['decimal'] - actual_result)
                accuracy_percent = max(0, (1024 - diff) / 1024 * 100)
                
                print(f"\n📊 비교 결과:")
                print(f"   AI 예측: {ai_result['decimal']}")
                print(f"   실제 값: {actual_result}")
                print(f"   차이: {diff} (정확도: {accuracy_percent:.1f}%)")
                
                if diff == 0:
                    print("🎉 완벽한 예측!")
                elif diff < 50:
                    print("👍 매우 좋은 예측")
                elif diff < 200:
                    print("👌 괜찮은 예측")
                else:
                    print("📈 예측 개선 필요")
            else:
                print("❌ 실제 시뮬레이션 실패")
        else:
            print("⚠️ C++ 라이브러리가 로드되지 않아 실제 시뮬레이션을 실행할 수 없습니다.")
    
    def interactive_library_load(self):
        """대화형 라이브러리 패턴 로드"""
        if not self.pattern_library:
            print("❌ 패턴 라이브러리가 없습니다.")
            return
        
        # 카테고리 선택
        categories = [k for k in self.pattern_library.keys() if not k.startswith('_')]
        if not categories:
            print("❌ 사용 가능한 패턴 카테고리가 없습니다.")
            return
        
        print(f"\n📚 패턴 카테고리 선택:")
        for i, category in enumerate(categories, 1):
            category_data = self.pattern_library[category]
            desc = category_data.get('_description', '')
            patterns_count = len([k for k in category_data.keys() if not k.startswith('_')])
            print(f"  {i}. {category} ({patterns_count}개) - {desc}")
        
        try:
            cat_choice = input(f"\n카테고리 선택 (1-{len(categories)}): ").strip()
            if not cat_choice:
                return
            
            cat_idx = int(cat_choice) - 1
            if not (0 <= cat_idx < len(categories)):
                print("❌ 잘못된 선택입니다.")
                return
            
            selected_category = categories[cat_idx]
            category_data = self.pattern_library[selected_category]
            patterns = {k: v for k, v in category_data.items() if not k.startswith('_')}
            
            if not patterns:
                print("❌ 해당 카테고리에 패턴이 없습니다.")
                return
            
            # 패턴 선택
            print(f"\n🎯 {selected_category} 패턴 선택:")
            pattern_list = list(patterns.items())
            
            for i, (pattern_name, pattern_data) in enumerate(pattern_list, 1):
                name = pattern_data.get('name', pattern_name)
                desc = pattern_data.get('description', '')[:50]
                period = pattern_data.get('period', '?')
                
                period_str = f"주기-{period}" if period > 0 else "불안정" if period == 0 else ""
                print(f"  {i:2d}. {pattern_name} - {name}")
                print(f"      {desc} ({period_str})")
            
            pattern_choice = input(f"\n패턴 선택 (1-{len(pattern_list)}): ").strip()
            if not pattern_choice:
                return
            
            pattern_idx = int(pattern_choice) - 1
            if not (0 <= pattern_idx < len(pattern_list)):
                print("❌ 잘못된 선택입니다.")
                return
            
            selected_pattern = pattern_list[pattern_idx][0]
            
            # 패턴 로드
            if self.load_library_pattern(selected_category, selected_pattern):
                self.print_grid(f"{patterns[selected_pattern].get('name', selected_pattern)}")
                
        except ValueError:
            print("❌ 숫자를 입력해주세요.")
        except KeyboardInterrupt:
            print("\n취소되었습니다.")
    
    def select_model_interactive(self):
        """대화형 모델 선택"""
        if not self.available_models:
            print("❌ 사용 가능한 모델이 없습니다.")
            return
        
        print(f"\n📋 사용 가능한 모델 ({len(self.available_models)}개):")
        for i, model_path in enumerate(self.available_models):
            model_name = os.path.basename(model_path)
            mod_time = datetime.fromtimestamp(os.path.getmtime(model_path))
            current = " [현재]" if model_path == self.current_model_path else ""
            print(f"  {i+1:2d}. {model_name} ({mod_time.strftime('%m-%d %H:%M')}){current}")
        
        try:
            choice = input(f"\n모델 선택 (1-{len(self.available_models)}): ").strip()
            if not choice:
                return
            
            idx = int(choice) - 1
            if 0 <= idx < len(self.available_models):
                selected_model = self.available_models[idx]
                self.load_model(selected_model)
            else:
                print("❌ 잘못된 선택입니다.")
        except ValueError:
            print("❌ 숫자를 입력해주세요.")
    
    def show_help(self):
        """도움말 출력"""
        help_text = """
🎮 Game of Life CLI Interface 도움말

=== 주요 명령어 ===
  edit     - 그리드 편집 모드 (대화형)
  predict  - 현재 그리드로 예측 테스트 실행
  show     - 현재 그리드 출력
  clear    - 그리드 초기화
  random   - 랜덤 그리드 생성

=== 패턴 관리 ===
  save     - 현재 패턴 저장
  load     - 패턴 로드 (사용자/라이브러리)
  patterns - 패턴 목록 보기 (사용자/라이브러리)

=== 모델 관리 ===
  model    - 모델 선택
  models   - 사용 가능한 모델 목록

=== 기타 ===
  help     - 이 도움말
  quit     - 프로그램 종료

=== 그리드 편집 모드 (edit 명령 후) ===
  x,y      - (x,y) 좌표 셀 토글 (예: 3,4)
  clear    - 그리드 지우기
  random   - 랜덤 패턴 생성
  show     - 그리드 보기
  done     - 편집 모드 종료

=== 예제 사용법 ===
  1. edit          # 그리드 편집
  2. 3,4           # (3,4) 셀 토글
  3. 5,6           # (5,6) 셀 토글
  4. done          # 편집 완료
  5. predict       # 예측 실행
  6. save          # 패턴 저장

🎯 팁: 'random' 명령으로 빠르게 테스트 패턴을 생성할 수 있습니다!
"""
        print(help_text)
    
    def run_interactive(self):
        """메인 대화형 루프"""
        print("🎮 Game of Life CLI Interface")
        print("   도움말: 'help' 입력 | 종료: 'quit' 입력")
        
        if self.model and self.current_model_path:
            model_name = os.path.basename(self.current_model_path)[:30]
            print(f"   현재 모델: {model_name}")
        else:
            print("   ⚠️ 모델이 로드되지 않았습니다")
        
        while True:
            try:
                command = input("\n> ").strip().lower()
                
                if command in ['quit', 'exit', 'q']:
                    print("👋 프로그램을 종료합니다.")
                    break
                elif command == 'help':
                    self.show_help()
                elif command == 'edit':
                    self.edit_grid_interactive()
                elif command == 'predict':
                    self.run_prediction_test()
                elif command == 'show':
                    self.print_grid()
                elif command == 'clear':
                    self.grid = np.zeros((10, 10), dtype=np.float32)
                    print("그리드가 초기화되었습니다.")
                elif command == 'random':
                    density = input("셀 밀도 (0.0-1.0, 기본값 0.3): ").strip()
                    try:
                        density = float(density) if density else 0.3
                        density = max(0.0, min(1.0, density))
                        self.grid = np.random.choice([0, 1], size=(10, 10), p=[1-density, density]).astype(np.float32)
                        print(f"랜덤 그리드 생성 (밀도: {density})")
                        self.print_grid()
                    except ValueError:
                        print("잘못된 밀도 값입니다.")
                elif command == 'save':
                    self.save_current_pattern()
                elif command == 'load':
                    print("\n패턴 로드 옵션:")
                    print("  1. 저장된 패턴 (사용자 생성)")
                    print("  2. 라이브러리 패턴 (정물, 진동자 등)")
                    
                    choice = input("선택 (1 또는 2): ").strip()
                    if choice == '1':
                        self.list_saved_patterns()
                        if self.saved_patterns:
                            pattern_name = input("로드할 패턴 이름: ").strip()
                            if pattern_name:
                                self.load_pattern(pattern_name)
                    elif choice == '2':
                        self.interactive_library_load()
                elif command == 'patterns':
                    print("\n패턴 목록 옵션:")
                    print("  1. 저장된 패턴 (사용자 생성)")
                    print("  2. 라이브러리 패턴 (정물, 진동자 등)")
                    
                    choice = input("선택 (1 또는 2): ").strip()
                    if choice == '1':
                        self.list_saved_patterns()
                    elif choice == '2':
                        self.list_pattern_library()
                elif command == 'model':
                    self.select_model_interactive()
                elif command == 'models':
                    if not self.available_models:
                        print("❌ 사용 가능한 모델이 없습니다.")
                    else:
                        print(f"\n📋 사용 가능한 모델 ({len(self.available_models)}개):")
                        for i, model_path in enumerate(self.available_models):
                            model_name = os.path.basename(model_path)
                            mod_time = datetime.fromtimestamp(os.path.getmtime(model_path))
                            current = " [현재]" if model_path == self.current_model_path else ""
                            print(f"  {i+1:2d}. {model_name} ({mod_time.strftime('%m-%d %H:%M')}){current}")
                elif command == '':
                    continue
                else:
                    print(f"❌ 알 수 없는 명령어: {command}")
                    print("   도움말: 'help' 입력")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Ctrl+C로 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 오류 발생: {e}")

def main():
    parser = argparse.ArgumentParser(description='Game of Life CLI Interface')
    parser.add_argument('--model', type=str, help='모델 파일 경로')
    parser.add_argument('--headless', action='store_true', help='자동 테스트 모드 (GUI 없음)')
    parser.add_argument('--test-pattern', type=str, help='테스트할 패턴 파일')
    
    args = parser.parse_args()
    
    # CLI 인터페이스 생성 및 실행
    cli = CLIGameOfLife(model_path=args.model, headless=args.headless)
    
    if args.headless:
        # 헤드리스 모드: 자동 테스트
        print("🤖 자동 테스트 모드")
        
        # 랜덤 패턴 생성 후 테스트
        cli.grid = np.random.choice([0, 1], size=(10, 10), p=[0.7, 0.3]).astype(np.float32)
        print("랜덤 테스트 패턴 생성")
        cli.print_grid("테스트 패턴")
        cli.run_prediction_test()
    else:
        # 대화형 모드
        cli.run_interactive()

if __name__ == "__main__":
    main()