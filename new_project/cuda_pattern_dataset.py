"""
CUDA 코드에 정의된 Game of Life 패턴들을 Python으로 구현한 데이터셋 생성기

패턴 정의들:
- oscillator_three_horizontal: Blinker 수평 (3셀)
- oscillator_three_vertical: Blinker 수직 (3셀)  
- oscillator_four: Block (2x2 정사각형)
- oscillator_five_*: 5셀 대각선 패턴들
- oscillator_six_*: Beehive 패턴들
- glider_*: 4방향 글라이더들
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
import os

class CUDAPatterns:
    """CUDA 코드에 정의된 패턴들을 Python으로 구현"""
    
    def __init__(self):
        self.patterns = {}
        self._define_cuda_patterns()
    
    def _define_cuda_patterns(self):
        """CUDA 코드의 패턴들을 그대로 구현 (패딩 포함)"""
        
        # oscillator_three_horizontal
        self.patterns['oscillator_three_horizontal'] = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]
        ])
        
        # oscillator_three_vertical
        self.patterns['oscillator_three_vertical'] = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 0]
        ])
        
        # oscillator_four (Block)
        self.patterns['oscillator_four'] = np.array([
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0]
        ])
        
        # oscillator_five_left_up
        self.patterns['oscillator_five_left_up'] = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ])
        
        # oscillator_five_left_down
        self.patterns['oscillator_five_left_down'] = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ])
        
        # oscillator_five_right_down
        self.patterns['oscillator_five_right_down'] = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0]
        ])
        
        # oscillator_five_right_up
        self.patterns['oscillator_five_right_up'] = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ])
        
        # oscillator_six_horizontal (Beehive 수평)
        self.patterns['oscillator_six_horizontal'] = np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])
        
        # oscillator_six_vertical (Beehive 수직)
        self.patterns['oscillator_six_vertical'] = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ])
        
        # glider_right_down
        self.patterns['glider_right_down'] = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]
        ])
        
        # glider_right_up
        self.patterns['glider_right_up'] = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ])
        
        # glider_left_down
        self.patterns['glider_left_down'] = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]
        ])
        
        # glider_left_up
        self.patterns['glider_left_up'] = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ])
    
    def get_pattern(self, name):
        """패턴 이름으로 패턴 배열 반환"""
        return self.patterns.get(name, None)
    
    def list_patterns(self):
        """모든 패턴 이름 리스트 반환"""
        return list(self.patterns.keys())
    
    def place_pattern_on_grid(self, pattern_name, grid_size=(10, 10), position=None):
        """
        패턴을 10x10 그리드에 배치
        
        Args:
            pattern_name: 패턴 이름
            grid_size: 그리드 크기 (항상 10x10)
            position: 배치 위치 (row, col), None이면 중앙
        
        Returns:
            10x10 그리드에 패턴이 배치된 numpy 배열
        """
        pattern = self.get_pattern(pattern_name)
        if pattern is None:
            raise ValueError(f"Unknown pattern: {pattern_name}")
        
        grid = np.zeros(grid_size, dtype=int)
        pattern_h, pattern_w = pattern.shape
        
        if position is None:
            # 중앙에 배치
            start_row = (grid_size[0] - pattern_h) // 2
            start_col = (grid_size[1] - pattern_w) // 2
        else:
            start_row, start_col = position
        
        # 그리드 경계 확인 및 클리핑
        end_row = min(start_row + pattern_h, grid_size[0])
        end_col = min(start_col + pattern_w, grid_size[1])
        
        # 음수 인덱스 처리
        actual_start_row = max(0, start_row)
        actual_start_col = max(0, start_col)
        
        # 패턴에서 실제로 복사할 영역 계산
        pattern_start_row = max(0, -start_row)
        pattern_start_col = max(0, -start_col)
        pattern_end_row = pattern_start_row + (end_row - actual_start_row)
        pattern_end_col = pattern_start_col + (end_col - actual_start_col)
        
        # 배치
        if actual_start_row < grid_size[0] and actual_start_col < grid_size[1]:
            grid[actual_start_row:end_row, actual_start_col:end_col] = \
                pattern[pattern_start_row:pattern_end_row, pattern_start_col:pattern_end_col]
        
        return grid

class CUDAPatternDataset(Dataset):
    """CUDA 코드 패턴들로 구성된 데이터셋"""
    
    def __init__(self, num_samples=1000, include_patterns=None, grid_size=(10, 10), 
                 random_positions=True, random_rotations=True, seed=None):
        """
        Args:
            num_samples: 생성할 샘플 수
            include_patterns: 포함할 패턴들의 리스트 (None이면 모든 패턴)
            grid_size: 그리드 크기 (항상 10x10)
            random_positions: 패턴을 랜덤한 위치에 배치할지 여부
            random_rotations: 패턴을 랜덤하게 회전시킬지 여부
            seed: 랜덤 시드 (재현 가능한 결과를 위해)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self.patterns_manager = CUDAPatterns()
        self.num_samples = num_samples
        self.grid_size = grid_size
        self.random_positions = random_positions
        self.random_rotations = random_rotations
        
        if include_patterns is None:
            self.include_patterns = self.patterns_manager.list_patterns()
        else:
            # 패턴 이름 검증
            valid_patterns = []
            for pattern in include_patterns:
                if pattern in self.patterns_manager.patterns:
                    valid_patterns.append(pattern)
                else:
                    print(f"⚠️ 알 수 없는 패턴: {pattern}")
            self.include_patterns = valid_patterns
        
        self.data = []
        self.labels = []
        self.pattern_info = []
        
        if len(self.include_patterns) == 0:
            raise ValueError("사용 가능한 패턴이 없습니다!")
        
        self._generate_dataset()
    
    def _rotate_pattern(self, pattern, rotations):
        """패턴을 90도씩 회전 (0, 1, 2, 3 = 0°, 90°, 180°, 270°)"""
        return np.rot90(pattern, k=rotations)
    
    def _generate_dataset(self):
        """데이터셋 생성"""
        print(f"🎮 CUDA 패턴 데이터셋 생성 시작...")
        print(f"포함된 패턴: {len(self.include_patterns)}개")
        print(f"생성할 샘플: {self.num_samples}개")
        print(f"랜덤 위치: {self.random_positions}")
        print(f"랜덤 회전: {self.random_rotations}")
        
        for i in range(self.num_samples):
            # 랜덤하게 패턴 선택
            pattern_name = random.choice(self.include_patterns)
            original_pattern = self.patterns_manager.get_pattern(pattern_name)
            
            # 랜덤 회전 적용
            rotation = 0
            if self.random_rotations:
                rotation = random.randint(0, 3)
                pattern = self._rotate_pattern(original_pattern, rotation)
            else:
                pattern = original_pattern.copy() if original_pattern is not None else np.zeros((3, 3))
            
            # 회전된 패턴 크기
            pattern_h, pattern_w = pattern.shape
            
            # 위치 결정
            if self.random_positions:
                # 그리드 안에 완전히 들어갈 수 있는 위치 범위 계산
                max_row = max(0, self.grid_size[0] - pattern_h)
                max_col = max(0, self.grid_size[1] - pattern_w)
                
                # 일부가 잘릴 수 있도록 범위를 확장 (더 다양한 패턴 생성)
                extended_max_row = self.grid_size[0] - 1
                extended_max_col = self.grid_size[1] - 1
                extended_min_row = -(pattern_h - 1)
                extended_min_col = -(pattern_w - 1)
                
                position = (
                    random.randint(extended_min_row, extended_max_row),
                    random.randint(extended_min_col, extended_max_col)
                )
            else:
                position = None  # 중앙 배치
            
            # 그리드에 패턴 배치
            grid = np.zeros(self.grid_size, dtype=int)
            
            # 수동으로 패턴 배치 (place_pattern_on_grid 대신)
            if position is None:
                start_row = (self.grid_size[0] - pattern_h) // 2
                start_col = (self.grid_size[1] - pattern_w) // 2
            else:
                start_row, start_col = position
            
            # 실제 배치
            for r in range(pattern_h):
                for c in range(pattern_w):
                    grid_r = start_row + r
                    grid_c = start_col + c
                    
                    if 0 <= grid_r < self.grid_size[0] and 0 <= grid_c < self.grid_size[1]:
                        grid[grid_r, grid_c] = pattern[r, c]
            
            # 패턴 이름을 숫자로 변환 (라벨)
            label = self._pattern_name_to_label(pattern_name)
            
            self.data.append(grid)
            self.labels.append(label)
            self.pattern_info.append({
                'pattern_name': pattern_name,
                'position': position,
                'rotation': rotation * 90,  # 각도로 저장
                'sample_index': i
            })
        
        print(f"✅ 데이터셋 생성 완료: {len(self.data)}개 샘플")
        print(f"라벨 범위: {min(self.labels)} ~ {max(self.labels)}")
    
    def _pattern_name_to_label(self, pattern_name):
        """패턴 이름을 숫자 라벨로 변환 (CUDA 코드 순서대로)"""
        pattern_to_id = {
            'oscillator_three_horizontal': 0,
            'oscillator_three_vertical': 1,
            'oscillator_four': 2,
            'oscillator_five_left_up': 3,
            'oscillator_five_right_down': 4,
            'oscillator_five_right_up': 5,
            'oscillator_five_left_down': 6,
            'oscillator_six_horizontal': 7,
            'oscillator_six_vertical': 8,
            'glider_left_up': 9,
            'glider_left_down': 10,
            'glider_right_up': 11,
            'glider_right_down': 12
        }
        return pattern_to_id.get(pattern_name, 0)
    
    def _label_to_pattern_name(self, label):
        """라벨을 패턴 이름으로 변환"""
        id_to_pattern = {
            0: 'oscillator_three_horizontal',
            1: 'oscillator_three_vertical',
            2: 'oscillator_four',
            3: 'oscillator_five_left_up',
            4: 'oscillator_five_right_down',
            5: 'oscillator_five_right_up',
            6: 'oscillator_five_left_down',
            7: 'oscillator_six_horizontal',
            8: 'oscillator_six_vertical',
            9: 'glider_left_up',
            10: 'glider_left_down',
            11: 'glider_right_up',
            12: 'glider_right_down'
        }
        return id_to_pattern.get(label, 'unknown')
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 10x10 패턴을 텐서로 변환
        data = torch.tensor(self.data[idx], dtype=torch.float32)
        
        # 10bit 변환 (라벨이 0-12이므로 충분)
        label_value = self.labels[idx]
        label_10bit = [(label_value >> i) & 1 for i in range(9, -1, -1)]
        label = torch.tensor(label_10bit, dtype=torch.float32)
        
        return data, label
    
    def save_to_file(self, filename):
        """데이터셋을 파일로 저장 (CUDA 데이터 형식과 호환)"""
        print(f"💾 데이터셋을 파일로 저장: {filename}")
        
        with open(filename, 'w', encoding='utf-8') as f:
            for i, (grid, label, info) in enumerate(zip(self.data, self.labels, self.pattern_info)):
                f.write(f"[{i}]\n")
                
                # 10x10 그리드 저장
                for row in grid:
                    f.write(''.join(map(str, row)) + '\n')
                
                # 라벨 저장
                f.write(f"{label}\n")
        
        print(f"✅ 저장 완료: {len(self.data)}개 샘플")
    
    def print_statistics(self):
        """데이터셋 통계 출력"""
        print("\n" + "="*60)
        print("📊 CUDA 패턴 데이터셋 통계")
        print("="*60)
        
        # 패턴별 샘플 수 계산
        pattern_counts = {}
        rotation_counts = {0: 0, 90: 0, 180: 0, 270: 0}
        
        for info in self.pattern_info:
            pattern_name = info['pattern_name']
            pattern_counts[pattern_name] = pattern_counts.get(pattern_name, 0) + 1
            rotation_counts[info['rotation']] += 1
        
        print(f"총 샘플 수: {len(self.data)}")
        print(f"그리드 크기: {self.grid_size}")
        print(f"라벨 범위: {min(self.labels)} ~ {max(self.labels)}")
        
        print("\n📋 패턴별 샘플 수:")
        for pattern_name, count in sorted(pattern_counts.items()):
            label = self._pattern_name_to_label(pattern_name)
            print(f"  {pattern_name:30s}: {count:4d}개 (라벨 {label:2d})")
        
        if self.random_rotations:
            print("\n🔄 회전별 샘플 수:")
            for angle, count in rotation_counts.items():
                print(f"  {angle:3d}°: {count:4d}개")
    
    def show_sample_patterns(self, num_samples=5):
        """샘플 패턴들을 텍스트로 출력"""
        print(f"\n🔍 샘플 패턴 미리보기 (처음 {num_samples}개):")
        print("="*60)
        
        for i in range(min(num_samples, len(self.data))):
            info = self.pattern_info[i]
            print(f"\n샘플 {i}: {info['pattern_name']} (라벨 {self.labels[i]}, 회전 {info['rotation']}°)")
            
            grid = self.data[i]
            for row in grid:
                print(''.join('█' if cell else '·' for cell in row))

def create_cuda_pattern_dataloader(num_samples=1000, include_patterns=None, 
                                 batch_size=32, shuffle=True, 
                                 random_positions=True, random_rotations=True, seed=None):
    """CUDA 패턴 데이터로더 생성"""
    dataset = CUDAPatternDataset(
        num_samples=num_samples,
        include_patterns=include_patterns,
        random_positions=random_positions,
        random_rotations=random_rotations,
        seed=seed
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
    
    return dataloader, dataset

def main():
    """메인 함수 - 데이터셋 생성 및 저장"""
    
    print("🎮 CUDA Game of Life 패턴 데이터셋 생성기")
    print("="*60)
    
    # 패턴 관리자 생성
    patterns = CUDAPatterns()
    
    print("사용 가능한 패턴들:")
    pattern_names = patterns.list_patterns()
    for i, pattern_name in enumerate(pattern_names):
        print(f"{i+1:2d}. {pattern_name}")
    
    # 데이터셋 생성 옵션
    num_samples = int(input(f"\n생성할 샘플 수를 입력하세요 (기본값: 2000): ") or "2000")
    
    # 특정 패턴만 사용할지 선택
    use_all = input("모든 패턴을 사용하시겠습니까? (Y/n): ").strip().lower()
    
    if use_all == 'n':
        print("\n사용할 패턴 번호들을 입력하세요 (쉼표로 구분, 예: 1,2,3):")
        selected_indices = input().strip().split(',')
        try:
            include_patterns = [pattern_names[int(i.strip())-1] for i in selected_indices if i.strip().isdigit()]
        except (IndexError, ValueError):
            print("잘못된 입력입니다. 모든 패턴을 사용합니다.")
            include_patterns = None
    else:
        include_patterns = None
    
    # 랜덤 위치 여부
    random_pos = input("패턴을 랜덤한 위치에 배치하시겠습니까? (Y/n): ").strip().lower()
    random_positions = random_pos != 'n'
    
    # 랜덤 회전 여부
    random_rot = input("패턴을 랜덤하게 회전시키시겠습니까? (Y/n): ").strip().lower()
    random_rotations = random_rot != 'n'
    
    # 시드 설정
    use_seed = input("고정된 시드를 사용하시겠습니까? (Y/n): ").strip().lower()
    if use_seed != 'n':
        seed = int(input("시드 값을 입력하세요 (기본값: 42): ") or "42")
    else:
        seed = None
    
    # 데이터셋 생성
    print("\n📊 데이터셋 생성 중...")
    dataset = CUDAPatternDataset(
        num_samples=num_samples,
        include_patterns=include_patterns,
        random_positions=random_positions,
        random_rotations=random_rotations,
        seed=seed
    )
    
    # 통계 출력
    dataset.print_statistics()
    
    # 샘플 패턴 미리보기
    show_preview = input("\n샘플 패턴을 미리보시겠습니까? (Y/n): ").strip().lower()
    if show_preview != 'n':
        dataset.show_sample_patterns(3)
    
    # 파일로 저장할지 선택
    save_file = input("\n데이터셋을 파일로 저장하시겠습니까? (Y/n): ").strip().lower()
    if save_file != 'n':
        filename = input("저장할 파일명을 입력하세요 (기본값: cuda_pattern_dataset.txt): ").strip()
        if not filename:
            filename = "cuda_pattern_dataset.txt"
        
        dataset.save_to_file(filename)
    
    # 데이터로더 테스트
    test_dataloader = input("\n데이터로더를 테스트하시겠습니까? (Y/n): ").strip().lower()
    if test_dataloader != 'n':
        print("\n🔍 데이터로더 테스트:")
        dataloader, _ = create_cuda_pattern_dataloader(
            num_samples=100,
            include_patterns=include_patterns,
            batch_size=16,
            random_positions=random_positions,
            random_rotations=random_rotations,
            seed=seed
        )
        
        for i, (batch_x, batch_y) in enumerate(dataloader):
            print(f"배치 {i+1}: 입력 크기 {batch_x.shape}, 라벨 크기 {batch_y.shape}")
            if i >= 2:  # 처음 3개 배치만 테스트
                break
        
        print("✅ 데이터로더 테스트 완료")

if __name__ == "__main__":
    main()