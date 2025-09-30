"""
데이터셋 JSON 설정 파일 로더
"""
import json
import os
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Any, List, Optional, Union
import numpy as np

class GameOfLifeDataset(Dataset):
    """Game of Life 시뮬레이션 데이터를 위한 Dataset 클래스"""
    
    def __init__(self, data_files):
        self.data = []
        self.labels = []
        
        for file_path in data_files:
            if os.path.exists(file_path):
                file_data, file_labels = self._load_file(file_path)
                self.data.extend(file_data)
                self.labels.extend(file_labels)
                print("✅ {}: {}개 샘플 로드".format(os.path.basename(file_path), len(file_data)))
            else:
                print("❌ 파일을 찾을 수 없습니다: {}".format(file_path))
        
        if len(self.data) == 0:
            print("❌ 로드된 데이터가 없습니다!")
        else:
            print("\n📊 총 로드된 샘플: {}개".format(len(self.data)))
            print("📊 전체 레이블 범위: {} ~ {}".format(min(self.labels), max(self.labels)))
    
    def _load_file(self, file_path):
        """단일 파일에서 데이터 로드 (형식: [n] + 10x10 패턴 + 레이블)"""
        file_data = []
        file_labels = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            i = 0
            
            while i < len(lines):
                line = lines[i].strip()
                
                # [숫자] 형식의 샘플 시작 확인
                if line.startswith('[') and line.endswith(']'):
                    try:
                        # 샘플 번호 추출
                        sample_num = int(line[1:-1])
                        
                        # 10x10 패턴 데이터 읽기
                        pattern_lines = []
                        for j in range(1, 11):  # 다음 10줄
                            if i + j < len(lines):
                                pattern_line = lines[i + j].strip()
                                if len(pattern_line) == 10 and all(c in '01' for c in pattern_line):
                                    pattern_lines.append([int(bit) for bit in pattern_line])
                                else:
                                    break
                        
                        # 레이블 읽기 (11번째 줄)
                        if i + 11 < len(lines) and len(pattern_lines) == 10:
                            label_line = lines[i + 11].strip()
                            if label_line.isdigit():
                                label = int(label_line)
                                
                                # 10x10을 50x50으로 확장 (중앙에 배치)
                                expanded_pattern = np.zeros((50, 50), dtype=np.float32)
                                start_row = (50 - 10) // 2  # 20
                                start_col = (50 - 10) // 2  # 20
                                
                                for row_idx, row in enumerate(pattern_lines):
                                    for col_idx, value in enumerate(row):
                                        expanded_pattern[start_row + row_idx, start_col + col_idx] = float(value)
                                
                                # 데이터 저장
                                file_data.append(expanded_pattern)
                                file_labels.append(label)
                        
                        i += 12  # 다음 샘플로 이동 ([n] + 10줄 패턴 + 1줄 레이블)
                    except (ValueError, IndexError):
                        i += 1
                else:
                    i += 1
        
        except Exception as e:
            print("파일 읽기 오류 {}: {}".format(file_path, e))
        
        return file_data, file_labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        grid = torch.FloatTensor(self.data[idx])
        label = self.labels[idx]
        
        # 레이블을 10비트 이진수로 변환
        binary_target = torch.zeros(10, dtype=torch.float32)
        for i in range(10):
            if label & (1 << (9-i)):
                binary_target[i] = 1.0
        
        return grid, binary_target

class DatasetLoader:
    """JSON 설정 파일을 사용한 데이터셋 로더"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = {}
        self.load_config()
    
    def _resolve_file_path(self, path: str) -> str:
        """파일 경로를 해결하는 단순화된 방식"""
        if os.path.isabs(path):
            # 절대 경로인 경우 그대로 사용
            return path
        
        # 상대 경로인 경우 3단계로만 시도
        # 1. 현재 작업 디렉토리 기준
        abs_path1 = os.path.join(os.getcwd(), path)
        if os.path.exists(abs_path1):
            return abs_path1
            
        # 2. config 파일 기준
        config_dir = os.path.dirname(os.path.abspath(self.config_path))
        abs_path2 = os.path.join(config_dir, path)
        if os.path.exists(abs_path2):
            return abs_path2
            
        # 3. 프로젝트 루트의 train_data 폴더에서 파일명으로 찾기
        filename = os.path.basename(path)
        current = os.getcwd()
        for _ in range(3):  # 최대 3단계 상위까지
            train_data_path = os.path.join(current, 'train_data', filename)
            if os.path.exists(train_data_path):
                return train_data_path
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
        
        # 파일을 찾지 못한 경우 첫 번째 경로를 기본값으로 반환
        print("❌ 파일을 찾을 수 없습니다: {}".format(path))
        return abs_path1
    
    def load_config(self):
        """JSON 설정 파일 로드"""
        try:
            if not os.path.exists(self.config_path):
                print("❌ 설정 파일을 찾을 수 없습니다: {}".format(self.config_path))
                return False
                
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
                
            print("✅ 설정 파일 로드 성공: {}".format(self.config_path))
            return True
            
        except json.JSONDecodeError as e:
            print("❌ JSON 파싱 오류: {}".format(e))
            return False
        except Exception as e:
            print("❌ 설정 파일 로드 실패: {}".format(e))
            return False
    
    def get_dataset_names(self) -> List[str]:
        """사용 가능한 데이터셋 이름 목록 반환"""
        if 'datasets' in self.config:
            return list(self.config['datasets'].keys())
        return []
    
    def get_dataset_info(self, dataset_name: str) -> Optional[Dict]:
        """데이터셋 정보 반환"""
        if 'datasets' in self.config and dataset_name in self.config['datasets']:
            return self.config['datasets'][dataset_name]
        return None
    
    def create_dataloader(self, dataset_name: str, batch_size: int = 32, 
                         shuffle: bool = True, num_workers: int = 4) -> Optional[DataLoader]:
        """데이터셋 로더 생성"""
        dataset_info = self.get_dataset_info(dataset_name)
        if not dataset_info:
            print("❌ 데이터셋 '{}'를 찾을 수 없습니다.".format(dataset_name))
            return None
        
        if dataset_info['type'] != 'simulation_files':
            print("❌ 지원하지 않는 데이터셋 타입: {}".format(dataset_info['type']))
            return None
        
        # 파일 경로 절대 경로로 변환 (단순화된 3단계 방식)
        file_paths = []
        
        for path in dataset_info['paths']:
            abs_path = self._resolve_file_path(path)
            file_paths.append(abs_path)
        
        print("📂 데이터셋 로딩: {} ({}개 파일)".format(dataset_info['name'], len(file_paths)))
        
        # 첫 번째 파일의 위치만 표시
        if file_paths and os.path.exists(file_paths[0]):
            print("📁 데이터 파일 위치: {}".format(os.path.dirname(file_paths[0])))
        
        # 데이터셋 생성
        dataset = GameOfLifeDataset(file_paths)
        
        if len(dataset) == 0:
            print("❌ 데이터셋이 비어있습니다.")
            return None
        
        # DataLoader 생성
        try:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True if torch.cuda.is_available() else False
            )
            print("🚀 DataLoader 생성 완료: batch_size={}, 총 배치 수={}".format(batch_size, len(dataloader)))
            return dataloader
            
        except Exception as e:
            print("❌ DataLoader 생성 실패: {}".format(e))
            return None
    
    def print_available_datasets(self):
        """사용 가능한 데이터셋 목록 출력"""
        dataset_names = self.get_dataset_names()
        if not dataset_names:
            print("❌ 사용 가능한 데이터셋이 없습니다.")
            return
        
        print("\n📊 사용 가능한 데이터셋:")
        for i, name in enumerate(dataset_names, 1):
            info = self.get_dataset_info(name)
            if info:
                print("  {}. {} - {}".format(i, name, info['name']))
                print("     예상 샘플: {:,}개".format(info['expected_samples']))
                print("     설명: {}".format(info['description']))
        print()

def load_dataset_from_files(file_paths: Union[str, List[str]], 
                           batch_size: int = 32, shuffle: bool = True, 
                           num_workers: int = 4) -> Optional[DataLoader]:
    """파일 경로로부터 직접 데이터셋 로드"""
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    
    # 파일 존재 확인
    missing_files = [path for path in file_paths if not os.path.exists(path)]
    if missing_files:
        print("❌ 다음 파일들을 찾을 수 없습니다:")
        for f in missing_files:
            print("   - {}".format(f))
        return None
    
    print("📂 직접 파일 로드: {}개 파일".format(len(file_paths)))
    
    # 데이터셋 생성
    dataset = GameOfLifeDataset(file_paths)
    if len(dataset) == 0:
        return None
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                     num_workers=num_workers, pin_memory=torch.cuda.is_available())