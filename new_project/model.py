import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from cuda_pattern_dataset import CUDAPatternDataset, create_cuda_pattern_dataloader

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 디바이스: {device}")

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

class DenseLayer(nn.Module):
    def __init__(self, inputLayer, hidden1Layer, hidden2Layer, hidden3Layer, outputLayer, activate):
        super(DenseLayer, self).__init__()
        self.fc1 = nn.Linear(inputLayer, hidden1Layer)
        self.fc2 = nn.Linear(hidden1Layer, hidden2Layer)
        self.fc3 = nn.Linear(hidden2Layer, hidden3Layer)
        self.fc4 = nn.Linear(hidden3Layer, outputLayer)
        self.fc_act = activate

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = getActive(self.fc_act, self.fc1(x))
        x = getActive(self.fc_act, self.fc2(x))
        x = getActive(self.fc_act, self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

class cnnLayer(nn.Module):
    def __init__(self, inputSize, hidden1Size, hidden2Size, outputSize, activate, stride, use_bias):
        super(cnnLayer, self).__init__()
        
        # CNN 레이어들 (3개)
        self.conv1 = nn.Conv2d(1, hidden1Size, kernel_size=3, stride=stride, padding=1, bias=use_bias)
        self.conv2 = nn.Conv2d(hidden1Size, hidden2Size, kernel_size=3, stride=stride, padding=1, bias=use_bias)
        self.conv3 = nn.Conv2d(hidden2Size, hidden2Size*2, kernel_size=3, stride=stride, padding=1, bias=use_bias)
        
        # 풀링 레이어
        self.pool = nn.MaxPool2d(2, 2)
        
        # 배치 정규화
        self.bn1 = nn.BatchNorm2d(hidden1Size, affine=False)
        self.bn2 = nn.BatchNorm2d(hidden2Size, affine=False)
        self.bn3 = nn.BatchNorm2d(hidden2Size*2, affine=False)
        
        # FC layer 입력 크기 계산
        self._calculate_fc_input_size(inputSize, hidden2Size*2)
        
        # Dense 레이어들 (5개)
        self.fc1 = nn.Linear(self.fc_input_size, 1024, bias=use_bias)
        self.fc2 = nn.Linear(1024, 512, bias=use_bias)
        self.fc3 = nn.Linear(512, 256, bias=use_bias)
        self.fc4 = nn.Linear(256, 128, bias=use_bias)
        self.fc5 = nn.Linear(128, outputSize, bias=use_bias)
        
        # 활성화 함수 저장
        self.fc_act = activate
    
    def _calculate_fc_input_size(self, inputSize, final_channels):
        """FC layer 입력 크기 계산"""
        # 10x10 → pool → 5x5 → pool → 2x2 → pool → 1x1
        # 3번의 pooling (2x2)으로 8배 감소
        size_after_conv = max(1, inputSize // 8)
        self.fc_input_size = final_channels * size_after_conv * size_after_conv
    
    def forward(self, x):
        # 입력 형태 조정: (batch, height, width) → (batch, 1, height, width)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # CNN 레이어들 (3개)
        x = self.pool(getActive(self.fc_act, self.bn1(self.conv1(x))))
        x = self.pool(getActive(self.fc_act, self.bn2(self.conv2(x))))
        x = self.pool(getActive(self.fc_act, self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Dense 레이어들 (5개)
        x = getActive(self.fc_act, self.fc1(x))
        x = getActive(self.fc_act, self.fc2(x))
        x = getActive(self.fc_act, self.fc3(x))
        x = getActive(self.fc_act, self.fc4(x))
        x = torch.sigmoid(self.fc5(x))  # 마지막 레이어는 활성화 함수 적용 안 함
        
        return x

def save_model(model, filepath, model_info=None, optimizer=None, epoch=0):
    """모델 가중치와 정보 저장 (옵티마이저 상태 포함)"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_info': model_info or {},
        'epoch': epoch
    }
    
    # 옵티마이저 상태도 저장 (추가 학습용)
    if optimizer is not None:
        save_dict['optimizer_state_dict'] = optimizer.state_dict()
    
    torch.save(save_dict, filepath)
    print(f"모델이 저장되었습니다: {filepath}")

def load_model(model_path, model_class, model_params, load_optimizer=False, lr=0.001):
    """모델 로드 (추가 학습 지원)"""
    try:
        if torch.cuda.is_available():
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(model_path, map_location='cpu')
        
        # 모델 정보 가져오기
        model_info = checkpoint.get('model_info', {})
        saved_epoch = checkpoint.get('epoch', 0)
        
        # 모델 생성
        model = model_class(**model_params).to(device)
        
        # 가중치 로드
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 옵티마이저 설정
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"옵티마이저 상태도 로드했습니다 (Epoch {saved_epoch}부터 계속)")
        
        print(f"모델 로드 성공: {model_path}")
        print(f"모델 정보: {model_info}")
        print(f"저장된 Epoch: {saved_epoch}")
        
        return model, optimizer, model_info, saved_epoch
        
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        return None, None, None, 0

def continue_training(model_path, train_loader, additional_epochs=100, lr=0.001, reset_optimizer=True, dataset_info=None):
    """기존 모델을 로드해서 추가 훈련"""
    
    # 모델 파라미터 정의 (저장된 정보에서 가져올 수도 있음)
    model_params = {
        'inputSize': 10,
        'hidden1Size': 32,
        'hidden2Size': 64,
        'outputSize': 10,
        'activate': 'swish',
        'stride': 1,
        'use_bias': False
    }
    
    # 모델 로드 (옵티마이저는 조건부)
    model, optimizer, model_info, start_epoch = load_model(
        model_path, cnnLayer, model_params, load_optimizer=not reset_optimizer, lr=lr
    )
    
    if model is None:
        print("모델 로드 실패.")
        return None, None
    
    # 옵티마이저 초기화 (새로운 학습률로)
    if reset_optimizer:
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        print(f"🔄 옵티마이저 새로 초기화 (학습률: {lr})")
    elif optimizer is None:
        # 옵티마이저 로드 실패 시 새로 생성
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        print(f"⚠️ 옵티마이저 로드 실패, 새로 생성 (학습률: {lr})")
    
    print(f"\n🔄 추가 훈련 시작: Epoch {start_epoch+1} ~ {start_epoch+additional_epochs}")
    
    # 추가 훈련
    criterion = nn.BCELoss()
    
    model.train()
    for epoch in range(additional_epochs):
        current_epoch = start_epoch + epoch + 1
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{current_epoch}/{start_epoch+additional_epochs}], Loss: {total_loss/len(train_loader):.4f}')
    
    # 업데이트된 모델 정보
    if model_info is not None:
        updated_info = model_info.copy()
        updated_info['total_epochs'] = start_epoch + additional_epochs
        updated_info['additional_training'] = True
        
        # 데이터셋 정보 업데이트 (전달받은 경우)
        if dataset_info is not None:
            updated_info['training_samples'] = dataset_info['samples']
            updated_info['label_range'] = dataset_info['label_range']
            updated_info['dataset_type'] = dataset_info['dataset_type']
            
    else:
        updated_info = {'total_epochs': start_epoch + additional_epochs, 'additional_training': True}
        if dataset_info is not None:
            updated_info.update(dataset_info)
    
    # 새 파일명으로 저장 (기존 파일 덮어쓰기 방지)
    new_model_path = model_path.replace('.pth', f'_continued_{start_epoch+additional_epochs}.pth')
    save_model(model, new_model_path, updated_info, optimizer, start_epoch + additional_epochs)
    
    return model, new_model_path

class NewBinaryDataset(Dataset):
    def __init__(self, data_file):
        self.data = []
        self.labels = []
        self.load_data(data_file)
    
    def load_data(self, data_file):
        """새로운 형식의 데이터 파일 로드"""
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            i = 0
            sample_count = 0
            
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
                                
                                # 데이터 저장
                                self.data.append(pattern_lines)
                                self.labels.append(label)
                                sample_count += 1
                        
                        i += 12  # 다음 샘플로 이동 ([n] + 10줄 패턴 + 1줄 레이블)
                        
                    except ValueError:
                        i += 1
                else:
                    i += 1
            
            print(f"Successfully loaded {sample_count} samples")
            if self.labels:
                print(f"Label range: {min(self.labels)} ~ {max(self.labels)}")
                
        except Exception as e:
            print(f"데이터 로딩 오류: {e}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 10x10 패턴을 텐서로 변환
        data = torch.tensor(self.data[idx], dtype=torch.float32)
        
        # 10bit 변환
        label_value = self.labels[idx]
        label_10bit = [(label_value >> i) & 1 for i in range(9, -1, -1)]
        label = torch.tensor(label_10bit, dtype=torch.float32)
        
        return data, label

def create_dataloader_from_file(data_file, batch_size=64, shuffle=True, num_workers=4):
    """파일에서 직접 데이터로더 생성"""
    dataset = NewBinaryDataset(data_file)
    
    if len(dataset) == 0:
        print("데이터셋이 비어있습니다.")
        return None, None
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    
    return dataloader, dataset

class BinaryDataset(Dataset):
    # ... 기존 코드 동일 ...
    def __init__(self, data_dir):
        self.data = []
        self.labels = []
        self.load_data(data_dir)
    
    def load_data(self, data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(data_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    if len(lines) < 11:
                        continue
                    
                    binary_patterns = []
                    for i in range(10):
                        line = lines[i].strip()
                        if line and all(c in '01' for c in line):
                            binary_pattern = [int(bit) for bit in line]
                            binary_patterns.append(binary_pattern)
                    
                    label_line = lines[10].strip()
                    if label_line.lstrip('-').isdigit():
                        label = int(label_line)
                    else:
                        continue
                    
                    if len(binary_patterns) == 10:
                        self.data.append(binary_patterns)
                        self.labels.append(label)
                        
                except Exception as e:
                    continue
        
        print(f"Successfully loaded {len(self.data)} samples")
        if self.labels:
            print(f"Label range: {min(self.labels)} ~ {max(self.labels)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx], dtype=torch.float32)
        
        label_value = self.labels[idx]
        label_10bit = [(label_value >> i) & 1 for i in range(9, -1, -1)]
        label = torch.tensor(label_10bit, dtype=torch.float32)
        
        return data, label

def create_dataloader(data_dir, batch_size=32, shuffle=True, num_workers=4):
    dataset = BinaryDataset(data_dir)
    if len(dataset) == 0:
        return None
    # num_workers 추가로 데이터 로딩 속도 향상
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                          num_workers=num_workers, pin_memory=True)
    return dataloader

def train_model(model, train_loader, epochs=100):
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            # 데이터를 GPU로 이동
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')


def predict_to_number(model, data_tensor):
    model.eval()
    with torch.no_grad():
        # 데이터를 GPU로 이동
        data_tensor = data_tensor.to(device)
        output = model(data_tensor)
        binary_output = (output > 0.5).float()
        # GPU에서 CPU로 이동 후 계산
        binary_output = binary_output.cpu()
        result = 0
        for i in range(10):
            result += int(binary_output[0][i]) * (2 ** (9-i))
        return result

class MultiFileDataset(Dataset):
    def __init__(self, file_list):
        self.data = []
        self.labels = []
        self.file_info = []
        self.load_multiple_files(file_list)
    
    def load_multiple_files(self, file_list):
        """여러 파일을 하나의 데이터셋으로 합침"""
        total_samples = 0
        
        for file_path in file_list:
            if os.path.exists(file_path):
                file_data, file_labels = self.load_single_file(file_path)
                
                # 파일별 통계
                if file_data:
                    survival_ratio = self.extract_survival_ratio(file_path)
                    self.file_info.append({
                        'path': file_path,
                        'samples': len(file_data),
                        'survival_ratio': survival_ratio,
                        'label_range': (min(file_labels), max(file_labels))
                    })
                    
                    self.data.extend(file_data)
                    self.labels.extend(file_labels)
                    total_samples += len(file_data)
                    
                    print(f"✅ {os.path.basename(file_path)}: {len(file_data)}개 샘플, 레이블 범위: {min(file_labels)}~{max(file_labels)}")
            else:
                print(f"❌ 파일 없음: {file_path}")
        
        print(f"\n📊 총 로드된 샘플: {total_samples}개")
        if self.labels:
            print(f"📊 전체 레이블 범위: {min(self.labels)} ~ {max(self.labels)}")
    
    def load_single_file(self, data_file):
        """단일 파일 로드"""
        data = []
        labels = []
        
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                if line.startswith('[') and line.endswith(']'):
                    try:
                        # 10x10 패턴 읽기
                        pattern_lines = []
                        for j in range(1, 11):
                            if i + j < len(lines):
                                pattern_line = lines[i + j].strip()
                                if len(pattern_line) == 10 and all(c in '01' for c in pattern_line):
                                    pattern_lines.append([int(bit) for bit in pattern_line])
                        
                        # 레이블 읽기
                        if i + 11 < len(lines) and len(pattern_lines) == 10:
                            label_line = lines[i + 11].strip()
                            if label_line.isdigit():
                                label = int(label_line)
                                data.append(pattern_lines)
                                labels.append(label)
                        
                        i += 12
                    except (ValueError, IndexError):
                        i += 1
                else:
                    i += 1
                    
        except Exception as e:
            print(f"파일 로드 오류 {data_file}: {e}")
        
        return data, labels
    
    def extract_survival_ratio(self, file_path):
        """파일명에서 생존 비율 추출"""
        try:
            # database-12345_100_0.030000.txt → 0.03
            parts = os.path.basename(file_path).split('_')
            ratio_part = parts[-1].replace('.txt', '')
            return float(ratio_part)
        except:
            return 0.0
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 10x10 패턴을 텐서로 변환
        data = torch.tensor(self.data[idx], dtype=torch.float32)
        
        # 10bit 변환
        label_value = self.labels[idx]
        label_10bit = [(label_value >> i) & 1 for i in range(9, -1, -1)]
        label = torch.tensor(label_10bit, dtype=torch.float32)
        
        return data, label
    
    def print_dataset_info(self):
        """데이터셋 정보 출력"""
        print("\n" + "="*60)
        print("📊 데이터셋 분석 결과")
        print("="*60)
        
        for info in self.file_info:
            print(f"생존비율 {info['survival_ratio']:.2f}: {info['samples']:3d}개, 레이블 {info['label_range'][0]:3d}~{info['label_range'][1]:3d}")
        
        print(f"\n총 파일 수: {len(self.file_info)}개")
        print(f"총 샘플 수: {len(self.data)}개")
        print(f"레이블 범위: {min(self.labels)} ~ {max(self.labels)}")

def create_multifile_dataloader(file_list, batch_size=64, shuffle=True, num_workers=4):
    """멀티 파일 데이터로더 생성"""
    dataset = MultiFileDataset(file_list)
    
    if len(dataset) == 0:
        print("❌ 데이터셋이 비어있습니다.")
        return None, None
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True)
    
    return dataloader, dataset



if __name__ == "__main__":
    # GPU 정보 출력
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    data_file = "/home/sjh100/바탕화면/pracPyTorch/train_data/database-12345_600000_0.300000.txt"

    data_file_testing = ["/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.010000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.020000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.030000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.040000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.050000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.060000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.070000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.080000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.090000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.100000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.110000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.120000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.130000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.140000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.150000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.160000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.170000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.180000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.190000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.200000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.210000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.220000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.230000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.240000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.250000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.260000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.270000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.280000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.290000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.300000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.310000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.320000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.330000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.340000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.350000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.360000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.370000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.380000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.390000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.400000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.410000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.420000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.430000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.440000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.450000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.460000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.470000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.480000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.490000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.500000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.510000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.520000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.530000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.540000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.550000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.560000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.570000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.580000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.590000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.600000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.610000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.620000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.630000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.640000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.650000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.660000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.670000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.680000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.690000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.700000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.710000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.720000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.730000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.740000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.750000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.760000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.770000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.780000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.790000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.800000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.810000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.820000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.830000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.840000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.850000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.860000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.870000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.880000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.890000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.900000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.910000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.920000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.930000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.940000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.950000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.960000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.970000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.980000.txt",
                         "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.990000.txt"]
    
    data_file_for_finetune = ["/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.010000.txt",
                              "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.020000.txt",
                              "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.030000.txt",
                              "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.040000.txt",
                              "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.050000.txt",
                              "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.060000.txt",
                              "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.070000.txt",
                              "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.080000.txt",
                              "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.090000.txt",
                              "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.100000.txt",
                              "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.110000.txt",
                              "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.120000.txt",
                              "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.130000.txt",
                              "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.140000.txt",
                              "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.150000.txt",
                              "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.160000.txt",
                              "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.170000.txt",
                              "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.180000.txt",
                              "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.190000.txt",
                              "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.200000.txt"]
    
    # 데이터로더 생성
    train_loader, dataset = create_multifile_dataloader(data_file_testing, batch_size=100)
    train_loader2, dataset2 = create_dataloader_from_file(data_file, batch_size=100)
    train_loader_fine, dataset_fine = create_dataloader_from_file(data_file_for_finetune, batch_size=100)
    
    # CUDA 패턴 데이터로더 (새로 추가)
    train_loader_patterns, dataset_patterns = create_cuda_pattern_dataloader(
        num_samples=2600,  # 13개 패턴 × 200개씩
        include_patterns=None,  # 모든 패턴 사용
        batch_size=100,
        random_positions=True,
        random_rotations=True,
        seed=42
    )
    
    if train_loader is None or dataset is None:
        print("데이터 로딩 실패")
        exit()
    print(f"총 샘플 수: {len(dataset)}")
    
    # 데이터셋 통계 출력
    if train_loader_patterns is not None and dataset_patterns is not None:
        print(f"\n🎮 CUDA 패턴 데이터셋 로드 성공: {len(dataset_patterns)}개 샘플")
        dataset_patterns.print_statistics()
    
    # 훈련 모드 선택
    mode = input("\n훈련 모드를 선택하세요:\n1. 새 모델 훈련\n2. 기존 모델 추가 훈련\n선택 (1 또는 2): ").strip()
    
    if mode == "2":
        # 기존 모델 추가 훈련
        model_path = input("기존 모델 파일 경로를 입력하세요 (예: saved_models/cnn_gol_model6.pth): ").strip()
        if not model_path:
            model_path = "saved_models/cnn_gol_model6_exp.pth"
        
        additional_epochs = int(input("추가 훈련할 에폭 수를 입력하세요 (기본값: 100): ") or "100")
        
        # 옵티마이저 초기화 여부 선택
        reset_opt = input("옵티마이저를 새로 초기화하시겠습니까? (Y/n): ").strip().lower()
        reset_optimizer = reset_opt != 'n'  # 기본값은 True
        
        # 새로운 학습률 설정 (옵티마이저 초기화 시)
        if reset_optimizer:
            new_lr = float(input("새 학습률을 입력하세요 (기본값: 0.0001): ") or "0.0001")
        else:
            new_lr = 0.001
        
        # 데이터셋 선택
        print("\n사용할 데이터셋을 선택하세요:")
        print("1. 10만개 대용량 데이터셋")
        print("2. 2만개 다양성 데이터셋")
        print("3. CUDA 패턴 데이터셋 (2600개)")
        
        dataset_choice = input("선택 (1, 2, 또는 3): ").strip()
        
        if dataset_choice == "1" and train_loader2 is not None and dataset2 is not None:
            selected_loader = train_loader2
            selected_dataset = dataset2
            print("🚀 60만개 대용량 데이터셋으로 추가 훈련")
        elif dataset_choice == "3" and train_loader_patterns is not None and dataset_patterns is not None:
            selected_loader = train_loader_patterns
            selected_dataset = dataset_patterns
            print("🎮 CUDA 패턴 데이터셋으로 추가 훈련")
        else:
            selected_loader = train_loader_fine
            selected_dataset = dataset_fine
            print("📊 2만개 다양성 데이터셋으로 추가 훈련")
        
        print(f"\n🔄 기존 모델 추가 훈련: {model_path}")
        
        # 데이터셋 정보 준비
        dataset_info = None
        if selected_dataset is not None:
            dataset_info = {
                'samples': len(selected_dataset),
                'label_range': (min(selected_dataset.labels), max(selected_dataset.labels)),
                'dataset_type': '대용량 60만개' if selected_dataset == dataset2 else '다양성 2800개'
            }
            print(f"📈 사용 데이터: {dataset_info['samples']}개 샘플, 레이블 범위: {dataset_info['label_range'][0]}~{dataset_info['label_range'][1]}")
        
        result = continue_training(model_path, selected_loader, additional_epochs, lr=new_lr, reset_optimizer=reset_optimizer, dataset_info=dataset_info)
        
        if result[0] is not None and result[1] is not None:
            model, new_path = result
            print(f"추가 훈련 완료! 새 모델 저장: {new_path}")
            
            # 테스트 예측
            test_data = dataset[0][0].unsqueeze(0)
            predicted_number = predict_to_number(model, test_data)
            print(f'테스트 - 실제: {dataset.labels[0]}, 예측: {predicted_number}')
        else:
            print("추가 훈련 실패!")
    
    else:
        # 새 모델 훈련
        print("\n🆕 새 모델 훈련 시작...")
        
        # 데이터셋 선택
        print("\n사용할 데이터셋을 선택하세요:")
        print("1. 9만개 테스트 데이터셋 (기본)")
        print("2. 60만개 대용량 데이터셋")
        print("3. CUDA 패턴 데이터셋 (2600개)")
        
        dataset_choice = input("선택 (1, 2, 또는 3): ").strip()
        
        if dataset_choice == "2" and train_loader2 is not None and dataset2 is not None:
            selected_train_loader = train_loader2
            selected_train_dataset = dataset2
            print("🚀 60만개 대용량 데이터셋으로 새 모델 훈련")
        elif dataset_choice == "3" and train_loader_patterns is not None and dataset_patterns is not None:
            selected_train_loader = train_loader_patterns
            selected_train_dataset = dataset_patterns
            print("🎮 CUDA 패턴 데이터셋으로 새 모델 훈련")
        else:
            selected_train_loader = train_loader
            selected_train_dataset = dataset
            print("📊 9만개 테스트 데이터셋으로 새 모델 훈련")
        
        model = cnnLayer(
            inputSize=10,
            hidden1Size=32,
            hidden2Size=64,
            outputSize=10,
            activate='swish',
            stride=1,
            use_bias=False
        ).to(device)
        
        epochs = int(input("훈련할 에폭 수를 입력하세요 (기본값: 500): ") or "500")
        
        import time
        start_time = time.time()
        
        # 옵티마이저 생성 (저장용)
        optimizer = optim.AdamW(model.parameters(), lr=0.001)
        
        # 선택된 데이터셋으로 훈련
        train_model(model, selected_train_loader, epochs=epochs)
        
        end_time = time.time()
        print(f"훈련 시간: {end_time - start_time:.2f}초")

        model_info = {
            'input_size': 10,
            'hidden1_size': 32,
            'hidden2_size': 64,
            'output_size': 10,
            'activate': 'swish',
            'stride': 1,
            'training_samples': len(selected_train_dataset),
            'training_epochs': epochs,
            'label_range': (min(selected_train_dataset.labels), max(selected_train_dataset.labels)),
            'dataset_type': 'CUDA패턴' if selected_train_dataset == dataset_patterns else '시뮬레이션'
        }
        
        # 모델 저장 (옵티마이저 상태 포함)
        model_path = "saved_models/cnn_gol_model9.pth"
        save_model(model, model_path, model_info, optimizer, epochs)
        
        # 테스트 예측
        test_data = selected_train_dataset[0][0].unsqueeze(0)
        predicted_number = predict_to_number(model, test_data)
        print(f'테스트 - 실제: {selected_train_dataset.labels[0]}, 예측: {predicted_number}')

