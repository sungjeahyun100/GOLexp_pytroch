# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("사용 디바이스: {}".format(device))

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
            print("옵티마이저 상태도 로드했습니다 (Epoch {}부터 계속)".format(saved_epoch))
        
        print("모델 로드 성공: {}".format(model_path))
        print("모델 정보: {}".format(model_info))
        print("저장된 Epoch: {}".format(saved_epoch))
        
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
    def __init__(self, data_file_list):
        self.data = []
        self.labels = []
        if isinstance(data_file_list, str):
            # 단일 파일인 경우
            self.load_single_file(data_file_list)
        else:
            # 파일 리스트인 경우
            self.load_multiple_files(data_file_list)
    
    def load_single_file(self, data_file):
        """단일 파일 로드"""
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
            
            print(f"✅ {os.path.basename(data_file)}: {sample_count}개 샘플 로드")
            
        except Exception as e:
            print(f"❌ 파일 로드 오류 {data_file}: {e}")
    
    def load_multiple_files(self, file_list):
        """여러 파일을 하나의 데이터셋으로 합침"""
        total_samples = 0
        
        for file_path in file_list:
            if os.path.exists(file_path):
                initial_count = len(self.data)
                self.load_single_file(file_path)
                added_count = len(self.data) - initial_count
                total_samples += added_count
            else:
                print(f"❌ 파일 없음: {file_path}")
        
        print(f"\n📊 총 로드된 샘플: {total_samples}개")
        if self.labels:
            print(f"📊 전체 레이블 범위: {min(self.labels)} ~ {max(self.labels)}")
    
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

def create_dataloader_from_files(data_files, batch_size=64, shuffle=True, num_workers=4):
    """파일(들)에서 데이터로더 생성"""
    dataset = NewBinaryDataset(data_files)
    
    if len(dataset) == 0:
        print("❌ 데이터셋이 비어있습니다.")
        return None, None
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    
    return dataloader, dataset

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

if __name__ == "__main__":
    # GPU 정보 출력
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 2만개 데이터셋 파일 경로 (생존비율 0.01~0.20)
    data_file_small = [
        "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.010000.txt",
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
        "/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_0.200000.txt"
    ]
    
    # 99만개 데이터셋 파일 경로 (생존비율 0.01~0.99)
    data_file_full = []
    for i in range(1, 100):  # 0.01부터 0.99까지
        ratio = i / 100.0
        filename = f"/home/sjh100/바탕화면/pracPyTorch/train_data/database-54321_1000_{ratio:.6f}.txt"
        data_file_full.append(filename)
    
    # 데이터셋 선택
    print("\n사용할 데이터셋을 선택하세요:")
    print("1. 소형 데이터셋 (2만개, 생존비율 0.01~0.20)")
    print("2. 전체 데이터셋 (9.9만개, 생존비율 0.01~0.99)")
    
    dataset_choice = input("선택 (1 또는 2): ").strip()
    
    if dataset_choice == "2":
        print("🚀 전체 데이터셋 로드 중...")
        train_loader, dataset = create_dataloader_from_files(data_file_full, batch_size=100)
        dataset_name = "99만개 전체 데이터셋"
    else:
        print("📊 소형 데이터셋 로드 중...")
        train_loader, dataset = create_dataloader_from_files(data_file_small, batch_size=100)
        dataset_name = "2만개 소형 데이터셋"
    
    if train_loader is None or dataset is None:
        print("❌ 데이터 로딩 실패")
        exit()

    # 훈련 모드 선택
    mode = input("\n훈련 모드를 선택하세요:\n1. 새 모델 훈련\n2. 기존 모델 추가 훈련\n선택 (1 또는 2): ").strip()
    
    if mode == "2":
        # 기존 모델 추가 훈련
        model_path = input("기존 모델 파일 경로를 입력하세요 (예: saved_models/cnn_gol_model9.pth): ").strip()
        if not model_path:
            model_path = "saved_models/cnn_gol_model9.pth"
        
        additional_epochs = int(input("추가 훈련할 에폭 수를 입력하세요 (기본값: 100): ") or "100")
        
        # 옵티마이저 초기화 여부 선택
        reset_opt = input("옵티마이저를 새로 초기화하시겠습니까? (Y/n): ").strip().lower()
        reset_optimizer = reset_opt != 'n'  # 기본값은 True
        
        # 새로운 학습률 설정 (옵티마이저 초기화 시)
        if reset_optimizer:
            new_lr = float(input("새 학습률을 입력하세요 (기본값: 0.0001): ") or "0.0001")
        else:
            new_lr = 0.001
        
        print(f"\n🔄 기존 모델 추가 훈련: {model_path}")
        
        # 데이터셋 정보 준비
        if dataset is not None:
            dataset_info = {
                'samples': len(dataset),
                'label_range': (min(dataset.labels), max(dataset.labels)),
                'dataset_type': dataset_name
            }
            print(f"📈 사용 데이터: {dataset_info['samples']}개 샘플, 레이블 범위: {dataset_info['label_range'][0]}~{dataset_info['label_range'][1]}")
        else:
            dataset_info = None
        
        result = continue_training(model_path, train_loader, additional_epochs, lr=new_lr, reset_optimizer=reset_optimizer, dataset_info=dataset_info)
        
        if result[0] is not None and result[1] is not None:
            model, new_path = result
            print(f"추가 훈련 완료! 새 모델 저장: {new_path}")
            
            # 테스트 예측
            if dataset is not None and len(dataset) > 0:
                test_data = dataset[0][0].unsqueeze(0)
                predicted_number = predict_to_number(model, test_data)
                print(f'테스트 - 실제: {dataset.labels[0]}, 예측: {predicted_number}')
        else:
            print("추가 훈련 실패!")
    
    else:
        # 새 모델 훈련
        print("\n🆕 새 모델 훈련 시작...")
        print(f"📊 {dataset_name} 사용")
        
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
        
        # 훈련
        train_model(model, train_loader, epochs=epochs)
        
        end_time = time.time()
        print(f"훈련 시간: {end_time - start_time:.2f}초")

        model_info = {
            'input_size': 10,
            'hidden1_size': 32,
            'hidden2_size': 64,
            'output_size': 10,
            'activate': 'swish',
            'stride': 1,
            'training_samples': len(dataset) if dataset else 0,
            'training_epochs': epochs,
            'label_range': (min(dataset.labels), max(dataset.labels)) if dataset else (0, 0),
            'dataset_type': dataset_name
        }
        
        # 모델 저장 (옵티마이저 상태 포함)
        if dataset_choice == "2":
            model_path = "saved_models/cnn_gol_model_full99k.pth"
        else:
            model_path = "saved_models/cnn_gol_model_small20k.pth"
        save_model(model, model_path, model_info, optimizer, epochs)
        
        # 테스트 예측
        if dataset is not None and len(dataset) > 0:
            test_data = dataset[0][0].unsqueeze(0)
            predicted_number = predict_to_number(model, test_data)
            print(f'테스트 - 실제: {dataset.labels[0]}, 예측: {predicted_number}')