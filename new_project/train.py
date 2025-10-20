#!/usr/bin/env python3
"""
Game of Life CNN 모델 훈련 스크립트 (간소화 버전)

사용법:
    python3 train.py --dataset small_simulation --epochs 50
    python3 train.py --files ../train_data/*.txt --epochs 20
"""
import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim

from src.data_loader import DatasetLoader, load_dataset_from_files
from src.model import CNNLayer, save_model, device
import json

def find_config_file(filename):
    """설정 파일을 찾는 단순화된 함수"""
    # 1. 현재 디렉토리
    if os.path.exists(filename):
        return filename
    
    # 2. 스크립트와 같은 디렉토리
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, filename)
    if os.path.exists(config_path):
        return config_path
    
    # 3. 기본값으로 현재 디렉토리의 파일명 반환
    return filename

def load_model_config(model_name):
    """모델 설정을 model_hyper.json에서 로드"""
    try:
        model_config_path = find_config_file("model_hyper.json")
        with open(model_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        if 'model_configs' not in config:
            print("❌ model_configs가 model_hyper.json에 없습니다.")
            return None
            
        if model_name not in config['model_configs']:
            print(f"❌ 모델 '{model_name}'을 찾을 수 없습니다.")
            print("사용 가능한 모델:")
            for name, info in config['model_configs'].items():
                print(f"  - {name}: {info['description']}")
            return None
            
        return config['model_configs'][model_name]
    except Exception as e:
        print(f"❌ 모델 설정 로드 실패: {e}")
        return None

def print_available_models():
    """사용 가능한 모델 목록 출력"""
    try:
        model_config_path = find_config_file("model_hyper.json")
        with open(model_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        if 'model_configs' not in config:
            print("❌ 모델 설정이 없습니다.")
            return
            
        print("\n📊 사용 가능한 모델 설정:")
        i = 1
        for name, info in config['model_configs'].items():
            if name.startswith('_'):
                continue
            print(f"  {i}. {name} - {info['name']}")
            print(f"     크기: {info['hidden1_size']}→{info['hidden2_size']}, 활성화: {info['activation']}")
            print(f"     추천: epochs={info['recommended_epochs']}, lr={info['recommended_lr']}, batch={info['recommended_batch_size']}")
            print(f"     설명: {info['description']}")
            i += 1
        print()
    except Exception as e:
        print(f"❌ 모델 목록 로드 실패: {e}")

def load_experiment_config(experiment_name):
    """실험 설정을 dataset_config.json에서 로드"""
    try:
        config_path = find_config_file("dataset_config.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        if 'custom_experiments' not in config:
            print("❌ custom_experiments가 설정 파일에 없습니다.")
            return None
            
        if experiment_name not in config['custom_experiments']:
            print(f"❌ 실험 '{experiment_name}'을 찾을 수 없습니다.")
            print("사용 가능한 실험:")
            for name, info in config['custom_experiments'].items():
                print(f"  - {name}: {info['description']}")
            return None
            
        return config['custom_experiments'][experiment_name]
    except Exception as e:
        print(f"❌ 실험 설정 로드 실패: {e}")
        return None

def print_available_experiments():
    """사용 가능한 실험 목록 출력"""
    try:
        config_path = find_config_file("dataset_config.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        if 'custom_experiments' not in config:
            print("❌ 실험 설정이 없습니다.")
            return
            
        print("\n🧪 사용 가능한 실험 설정:")
        i = 1
        for name, info in config['custom_experiments'].items():
            if name.startswith('_'):
                continue
            print(f"  {i}. {name} - {info['name']}")
            print(f"     데이터셋: {info['dataset']}, 모델: {info['model']}")
            print(f"     훈련: epochs={info['training_params']['epochs']}, lr={info['training_params']['learning_rate']}")
            print(f"     설명: {info['description']}")
            print(f"     노트: {info['notes']}")
            i += 1
        print()
    except Exception as e:
        print(f"❌ 실험 목록 로드 실패: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description='Game of Life CNN 훈련')
    
    # 데이터 옵션 (--list-models일 때는 optional)
    data_group = parser.add_mutually_exclusive_group(required=False)
    data_group.add_argument('--dataset', type=str, 
                           help='JSON 설정의 데이터셋 이름 (예: small_simulation)')
    data_group.add_argument('--files', nargs='+', 
                           help='직접 지정할 데이터 파일들')
    
    # 훈련 파라미터
    parser.add_argument('--epochs', type=int, default=50, help='에포크 수')
    parser.add_argument('--batch-size', type=int, default=32, help='배치 크기')
    parser.add_argument('--lr', type=float, default=0.001, help='학습률')
    
    # 실험 설정 옵션
    exp_group = parser.add_mutually_exclusive_group()
    exp_group.add_argument('--experiment', type=str,
                          help='JSON 설정의 실험 이름 (예: my_experiment_1)')
    exp_group.add_argument('--model-config', type=str, 
                           help='JSON 설정의 모델 이름 (예: standard_cnn)')
    exp_group.add_argument('--custom-model', action='store_true',
                           help='수동 모델 파라미터 사용')
    
    # 수동 모델 파라미터 (--custom-model과 함께 사용)
    parser.add_argument('--activation', default='swish', help='활성화 함수')
    parser.add_argument('--hidden1', type=int, default=32, help='첫 번째 은닉층 크기')
    parser.add_argument('--hidden2', type=int, default=64, help='두 번째 은닉층 크기')
    parser.add_argument('--use-bias', action='store_true', help='bias 사용 여부')
    parser.add_argument('--stride', type=int, default=1, help='stride 값')
    
    # 기타
    parser.add_argument('--output', type=str, help='모델 저장 경로')
    parser.add_argument('--quiet', action='store_true', help='최소 출력')
    parser.add_argument('--list-models', action='store_true', help='사용 가능한 모델 목록 표시')
    parser.add_argument('--list-experiments', action='store_true', help='사용 가능한 실험 목록 표시')
    
    return parser.parse_args()

def train_model(model, dataloader, epochs, lr, quiet=False, exp_dir=None):
    """모델 훈련 및 로스 기록"""
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # 로깅 파일 준비
    epoch_loss_file = None
    batch_loss_file = None
    if exp_dir:
        try:
            epoch_loss_path = os.path.join(exp_dir, "epoch-loss.txt")
            batch_loss_path = os.path.join(exp_dir, "batch-loss.txt")
            epoch_loss_file = open(epoch_loss_path, 'w', encoding='utf-8')
            batch_loss_file = open(batch_loss_path, 'w', encoding='utf-8')
            # 헤더 작성
            epoch_loss_file.write("epoch,avg_loss\n")
            batch_loss_file.write("epoch,batch_num,loss\n")
        except IOError as e:
            print(f"⚠️ 로깅 파일 열기 실패: {e}")
            exp_dir = None # 로깅 비활성화

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        
        for i, (batch_x, batch_y) in enumerate(dataloader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            batch_loss = loss.item()
            total_loss += batch_loss
            
            # 배치 로스 기록
            if batch_loss_file:
                batch_loss_file.write(f"{epoch+1},{i+1},{batch_loss:.6f}\n")
        
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        
        # 에폭 로스 기록
        if epoch_loss_file:
            epoch_loss_file.write(f"{epoch+1},{avg_loss:.6f}\n")

        if not quiet and (epoch + 1) % 10 == 0:
            print("Epoch [{}/{}], Loss: {:.4f}".format(epoch+1, epochs, avg_loss))
    
    # 파일 닫기
    if epoch_loss_file:
        epoch_loss_file.close()
    if batch_loss_file:
        batch_loss_file.close()

    return optimizer

def main():
    args = parse_args()
    
    # 목록 표시 후 종료
    if args.list_models:
        print_available_models()
        return 0
    
    if args.list_experiments:
        print_available_experiments()
        return 0
    
    # 실험 설정 처리
    if args.experiment:
        experiment_config = load_experiment_config(args.experiment)
        if experiment_config is None:
            print_available_experiments()
            return 1
        
        # 실험 설정에서 파라미터 추출
        args.dataset = experiment_config['dataset']
        args.model_config = experiment_config['model']
        
        # 훈련 파라미터 적용 (명령행에서 명시적으로 제공되지 않은 경우만)
        import sys
        
        # 명령행 인자에서 명시적으로 제공되었는지 확인
        epochs_provided = '--epochs' in sys.argv
        lr_provided = '--lr' in sys.argv
        batch_size_provided = '--batch-size' in sys.argv or '--batch_size' in sys.argv
        
        if not epochs_provided:
            args.epochs = experiment_config['training_params']['epochs']
            print(f"📊 실험 설정 epochs 적용: {args.epochs}")
        else:
            print(f"📊 명령행 epochs 사용: {args.epochs}")
            
        if not lr_provided:
            args.lr = experiment_config['training_params']['learning_rate']
            
        if not batch_size_provided:
            args.batch_size = experiment_config['training_params']['batch_size']
        
        if not args.quiet:
            print(f"🧪 실험 설정: {experiment_config['name']}")
            print(f"   설명: {experiment_config['description']}")
            print(f"   데이터셋: {experiment_config['dataset']}")
            print(f"   모델: {experiment_config['model']}")
            print(f"   노트: {experiment_config['notes']}")
    
    # 일반 훈련 시에는 데이터셋이 필수
    if not args.dataset and not args.files:
        print("❌ 오류: 훈련을 위해서는 --dataset, --files, 또는 --experiment가 필요합니다.")
        print("💡 사용법:")
        print("   python3 train.py --experiment my_experiment_1")
        print("   python3 train.py --dataset custom_experiment_1 --model-config small_cnn")
        print("   python3 train.py --list-experiments  # 실험 목록 확인")
        print("   python3 train.py --list-models  # 모델 목록 확인")
        return 1
    
    if not args.quiet:
        print("🚀 Game of Life CNN 훈련 시작")
        if torch.cuda.is_available():
            print("GPU: {}".format(torch.cuda.get_device_name(0)))
    
    # 데이터 로딩
    if args.dataset:
        # JSON 설정 사용 - config 파일 경로 자동 탐지
        config_path = find_config_file("dataset_config.json")
        loader = DatasetLoader(config_path)
        dataloader = loader.create_dataloader(
            args.dataset, 
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4
        )
        if dataloader is None:
            print("❌ 데이터셋 로딩 실패")
            return 1
        dataset_name = args.dataset
        
    else:
        # 직접 파일 지정
        dataloader = load_dataset_from_files(
            args.files,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4
        )
        if dataloader is None:
            print("❌ 파일 로딩 실패")
            return 1
        dataset_name = "custom_files"
    
    # 모델 설정 결정
    if args.model_config:
        # JSON 설정 사용
        config_path = find_config_file("dataset_config.json")
        model_config = load_model_config(args.model_config)
        if model_config is None:
            print_available_models()
            return 1
            
        # 추천 하이퍼파라미터 적용 (명령행에서 명시적으로 제공되지 않은 경우만)
        import sys
        
        epochs_provided = '--epochs' in sys.argv
        lr_provided = '--lr' in sys.argv
        batch_size_provided = '--batch-size' in sys.argv or '--batch_size' in sys.argv
        
        if not epochs_provided:
            args.epochs = model_config['recommended_epochs']
            print(f"📊 모델 추천 epochs 적용: {args.epochs}")
        else:
            print(f"📊 명령행 epochs 사용: {args.epochs}")
            
        if not lr_provided:
            args.lr = model_config['recommended_lr']
            
        if not batch_size_provided:
            args.batch_size = model_config['recommended_batch_size']
            # 이미 생성된 dataloader가 있다면 재생성 필요
            if args.dataset:
                dataloader = loader.create_dataloader(
                    args.dataset, 
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=4
                )
        
        model_name = args.model_config
        if not args.quiet:
            print(f"🎯 모델 설정: {model_config['name']}")
            print(f"   {model_config['description']}")
            print(f"   구조: {model_config['hidden1_size']}→{model_config['hidden2_size']}")
            print(f"   활성화: {model_config['activation']}, bias: {model_config['use_bias']}")
    else:
        # 수동 설정 사용 (기본값 또는 명령행 인자)
        model_config = {
            'input_size': 10,
            'hidden1_size': args.hidden1,
            'hidden2_size': args.hidden2,
            'output_size': 10,
            'activation': args.activation,
            'stride': args.stride,
            'use_bias': args.use_bias
        }
        model_name = "custom"
        if not args.quiet:
            print("🔧 커스텀 모델 설정 사용")

    # 모델 생성
    model = CNNLayer(
        input_size=model_config['input_size'],
        hidden1_size=model_config['hidden1_size'],
        hidden2_size=model_config['hidden2_size'],
        output_size=model_config['output_size'],
        activate=model_config['activation'],
        stride=model_config['stride'],
        use_bias=model_config['use_bias']
    ).to(device)
    
    if not args.quiet:
        total_params = sum(p.numel() for p in model.parameters())
        print("📊 모델 파라미터 수: {:,}".format(total_params))
        if dataloader is not None:
            print("📊 데이터셋 배치 수: {}".format(len(dataloader)))
    
    # 로깅 및 실험 ID 설정
    if args.experiment:
        exp_id = args.experiment
    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        exp_id = f"{timestamp}_{dataset_name}_{model_name}"

    exp_dir = os.path.join("graph", exp_id)
    if not args.quiet:
        print(f"📈 로스 기록 경로: {exp_dir}")
    os.makedirs(exp_dir, exist_ok=True)

    # 훈련
    start_time = time.time()
    optimizer = train_model(model, dataloader, args.epochs, args.lr, args.quiet, exp_dir=exp_dir)
    end_time = time.time()
    
    if not args.quiet:
        print("✅ 훈련 완료! 소요 시간: {:.1f}초".format(end_time - start_time))
    
    # 모델 저장
    if args.output:
        save_path = args.output
    else:
        save_path = "saved_models/gol_cnn_{}_{}.pth".format(dataset_name, model_name)
    
    # 모델 정보에 실제 사용된 설정 저장
    model_info = model_config.copy()
    model_info.update({
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'dataset': dataset_name,
        'model_config_name': model_name,
        'batch_size': args.batch_size
    })
    
    save_model(model, save_path, model_info, optimizer, args.epochs)
    
    if not args.quiet:
        print("🎯 모델 저장 완료: {}".format(save_path))
    
    return 0

if __name__ == "__main__":
    sys.exit(main())