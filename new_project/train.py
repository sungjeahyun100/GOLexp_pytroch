#!/usr/bin/env python3
"""
Game of Life CNN 모델 훈련 스크립트 (간소화 버전)

사용법:
    python3 train.py --dataset small_simulation --epochs 50
    python3 train.py --files ../train_data/*.txt --epochs 20
"""
import argparse
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim

from src.data_loader import DatasetLoader, load_dataset_from_files
from src.model import CNNLayer, save_model, device

def parse_args():
    parser = argparse.ArgumentParser(description='Game of Life CNN 훈련')
    
    # 데이터 옵션
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument('--dataset', type=str, 
                           help='JSON 설정의 데이터셋 이름 (예: small_simulation)')
    data_group.add_argument('--files', nargs='+', 
                           help='직접 지정할 데이터 파일들')
    
    # 훈련 파라미터
    parser.add_argument('--epochs', type=int, default=50, help='에포크 수')
    parser.add_argument('--batch-size', type=int, default=32, help='배치 크기')
    parser.add_argument('--lr', type=float, default=0.001, help='학습률')
    
    # 모델 파라미터
    parser.add_argument('--activation', default='swish', help='활성화 함수')
    parser.add_argument('--hidden1', type=int, default=32, help='첫 번째 은닉층 크기')
    parser.add_argument('--hidden2', type=int, default=64, help='두 번째 은닉층 크기')
    
    # 기타
    parser.add_argument('--output', type=str, help='모델 저장 경로')
    parser.add_argument('--quiet', action='store_true', help='최소 출력')
    
    return parser.parse_args()

def train_model(model, dataloader, epochs, lr, quiet=False):
    """모델 훈련"""
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        if not quiet and (epoch + 1) % 10 == 0:
            avg_loss = total_loss / batch_count if batch_count > 0 else 0
            print("Epoch [{}/{}], Loss: {:.4f}".format(epoch+1, epochs, avg_loss))
    
    return optimizer

def main():
    args = parse_args()
    
    if not args.quiet:
        print("🚀 Game of Life CNN 훈련 시작")
        if torch.cuda.is_available():
            print("GPU: {}".format(torch.cuda.get_device_name(0)))
    
    # 데이터 로딩
    if args.dataset:
        # JSON 설정 사용
        loader = DatasetLoader("dataset_config.json")
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
    
    # 모델 생성
    model = CNNLayer(
        input_size=50,
        hidden1_size=args.hidden1,
        hidden2_size=args.hidden2,
        output_size=10,
        activate=args.activation,
        stride=1,
        use_bias=False
    ).to(device)
    
    if not args.quiet:
        total_params = sum(p.numel() for p in model.parameters())
        print("📊 모델 파라미터 수: {:,}".format(total_params))
        print("📊 데이터셋 배치 수: {}".format(len(dataloader)))
    
    # 훈련
    start_time = time.time()
    optimizer = train_model(model, dataloader, args.epochs, args.lr, args.quiet)
    end_time = time.time()
    
    if not args.quiet:
        print("✅ 훈련 완료! 소요 시간: {:.1f}초".format(end_time - start_time))
    
    # 모델 저장
    if args.output:
        save_path = args.output
    else:
        save_path = "saved_models/gol_cnn_{}.pth".format(dataset_name)
    
    model_info = {
        'input_size': 50,
        'hidden1_size': args.hidden1,
        'hidden2_size': args.hidden2,
        'output_size': 10,
        'activate': args.activation,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'dataset': dataset_name
    }
    
    save_model(model, save_path, model_info, optimizer, args.epochs)
    
    if not args.quiet:
        print("🎯 모델 저장 완료: {}".format(save_path))
    
    return 0

if __name__ == "__main__":
    sys.exit(main())