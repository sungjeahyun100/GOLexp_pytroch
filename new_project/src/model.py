"""
Game of Life CNN 모델 정의
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_activation(activation_name, input_tensor):
    """활성화 함수 선택"""
    if activation_name == 'tanh':
        return F.tanh(input_tensor)
    elif activation_name == 'relu':
        return F.relu(input_tensor)
    elif activation_name == 'sigmoid':
        return F.sigmoid(input_tensor)
    elif activation_name == 'elu':
        return F.elu(input_tensor)
    elif activation_name == 'selu':
        return F.selu(input_tensor)
    elif activation_name == 'lrelu':
        return F.leaky_relu(input_tensor)
    elif activation_name == 'swish':
        return F.silu(input_tensor)
    else:
        return F.tanh(input_tensor)

class DenseLayer(nn.Module):
    """완전 연결 레이어만으로 구성된 네트워크"""
    
    def __init__(self, input_layer, hidden1_layer, hidden2_layer, hidden3_layer, output_layer, activate):
        super(DenseLayer, self).__init__()
        self.fc1 = nn.Linear(input_layer, hidden1_layer)
        self.fc2 = nn.Linear(hidden1_layer, hidden2_layer)
        self.fc3 = nn.Linear(hidden2_layer, hidden3_layer)
        self.fc4 = nn.Linear(hidden3_layer, output_layer)
        self.fc_act = activate

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = get_activation(self.fc_act, self.fc1(x))
        x = get_activation(self.fc_act, self.fc2(x))
        x = get_activation(self.fc_act, self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

class CNNLayer(nn.Module):
    """CNN + Dense 레이어로 구성된 네트워크 (Game of Life 예측용)"""
    
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, activate, stride, use_bias):
        super(CNNLayer, self).__init__()
        
        # CNN 레이어들 (3개)
        self.conv1 = nn.Conv2d(1, hidden1_size, kernel_size=3, stride=stride, padding=1, bias=use_bias)
        self.conv2 = nn.Conv2d(hidden1_size, hidden2_size, kernel_size=3, stride=stride, padding=1, bias=use_bias)
        self.conv3 = nn.Conv2d(hidden2_size, hidden2_size*2, kernel_size=3, stride=stride, padding=1, bias=use_bias)
        
        # 풀링 레이어
        self.pool = nn.MaxPool2d(2, 2)
        
        # 배치 정규화 (affine=False로 정보 보존)
        self.bn1 = nn.BatchNorm2d(hidden1_size, affine=False)
        self.bn2 = nn.BatchNorm2d(hidden2_size, affine=False)
        self.bn3 = nn.BatchNorm2d(hidden2_size*2, affine=False)
        
        # FC layer 입력 크기 계산
        self._calculate_fc_input_size(input_size, hidden2_size*2)
        
        # Dense 레이어들 (5개)
        self.fc1 = nn.Linear(self.fc_input_size, 1024, bias=use_bias)
        self.fc2 = nn.Linear(1024, 512, bias=use_bias)
        self.fc3 = nn.Linear(512, 256, bias=use_bias)
        self.fc4 = nn.Linear(256, 128, bias=use_bias)
        self.fc5 = nn.Linear(128, output_size, bias=use_bias)
        
        # 활성화 함수 저장
        self.fc_act = activate
    
    def _calculate_fc_input_size(self, input_size, final_channels):
        """FC layer 입력 크기 계산"""
        # 3번의 pooling (2x2)으로 8배 감소: 50x50 → 6x6
        size_after_conv = max(1, input_size // 8)
        self.fc_input_size = final_channels * size_after_conv * size_after_conv
    
    def forward(self, x):
        # 입력 형태 조정: (batch, height, width) → (batch, 1, height, width)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # CNN 레이어들 (3개)
        x = self.pool(get_activation(self.fc_act, self.bn1(self.conv1(x))))
        x = self.pool(get_activation(self.fc_act, self.bn2(self.conv2(x))))
        x = self.pool(get_activation(self.fc_act, self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Dense 레이어들 (5개)
        x = get_activation(self.fc_act, self.fc1(x))
        x = get_activation(self.fc_act, self.fc2(x))
        x = get_activation(self.fc_act, self.fc3(x))
        x = get_activation(self.fc_act, self.fc4(x))
        x = torch.sigmoid(self.fc5(x))  # 10비트 출력을 위한 시그모이드
        
        return x

def save_model(model, filepath, model_info=None, optimizer=None, epoch=0):
    """모델 가중치와 정보 저장 (옵티마이저 상태 포함)"""
    import os
    
    # 디렉토리 생성
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 저장할 딕셔너리 생성
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_info': model_info or {},
        'epoch': epoch
    }
    
    # 옵티마이저 상태도 저장 (추가 학습용)
    if optimizer is not None:
        save_dict['optimizer_state_dict'] = optimizer.state_dict()
    
    torch.save(save_dict, filepath)
    print("모델이 저장되었습니다: {}".format(filepath))

def load_model(model_path, model_class, model_params, load_optimizer=False, lr=0.001):
    """모델 로드 (추가 학습 지원)"""
    try:
        if torch.cuda.is_available():
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(model_path, map_location='cpu')
        
        # 모델 생성 및 가중치 로드
        model = model_class(**model_params)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        # 옵티마이저 로드 (선택사항)
        optimizer = None
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            import torch.optim as optim
            optimizer = optim.AdamW(model.parameters(), lr=lr)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print("모델 로드 성공: {}".format(model_path))
        print("저장된 에포크: {}".format(checkpoint.get('epoch', '알 수 없음')))
        
        if load_optimizer:
            return model, optimizer, checkpoint.get('model_info', {})
        else:
            return model, checkpoint.get('model_info', {})
            
    except FileNotFoundError:
        print("모델 파일을 찾을 수 없습니다: {}".format(model_path))
        return None
    except Exception as e:
        print("모델 로드 실패: {}".format(e))
        return None

def predict_to_number(model, input_data):
    """모델 예측을 숫자로 변환"""
    model.eval()
    with torch.no_grad():
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.to(device)
        
        prediction = model(input_data)
        binary_prediction = (prediction > 0.5).float()
        
        # 10비트 이진 출력을 숫자로 변환
        number = 0
        for i in range(10):
            if binary_prediction[0][i] > 0.5:
                number += 2 ** (9 - i)
        
        return int(number)