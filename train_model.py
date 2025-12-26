import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import os

WINDOW_SIZE = 5      
HIDDEN_SIZE = 128    
LAYERS = 2           
LEARNING_RATE = 0.001
EPOCHS = 200         

# 1. 데이터 전처리 클래스
class LottoDataset(Dataset):
    def __init__(self, data):
        self.x_data = []
        self.y_data = []
        
        # 슬라이딩 윈도우로 데이터 자르기
        for i in range(len(data) - WINDOW_SIZE):
            x = data[i : i + WINDOW_SIZE] 
            y = data[i + WINDOW_SIZE]     
            
            self.x_data.append(x)
            self.y_data.append(y)
            
        self.x_data = torch.tensor(np.array(self.x_data), dtype=torch.float32)
        self.y_data = torch.tensor(np.array(self.y_data), dtype=torch.float32)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

class LottoLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LottoLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size) 

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM 통과
        out, _ = self.lstm(x, (h0, c0))
        
        # 마지막 타임스텝의 결과만 가져오기
        out = self.fc(out[:, -1, :])
        return out

# --- 2. 메인 실행 코드 ---
if __name__ == "__main__":
    # 데이터 불러오기
    if not os.path.exists("lotto_history.csv"):
        print("❌ 데이터 파일이 없어요! get_data.py 먼저 실행하세요.")
        exit()
        
    df = pd.read_csv("lotto_history.csv")
    
    # 필요한 번호만 가져오기
    numbers = df[['drwtNo1', 'drwtNo2', 'drwtNo3', 'drwtNo4', 'drwtNo5', 'drwtNo6']].values
    
    scaler = MinMaxScaler()
    numbers_scaled = scaler.fit_transform(numbers)
    
    # 데이터 나누기 (1~1000회: 학습용 / 1001~끝: 검증용)
    # 주의: WINDOW_SIZE 만큼 데이터가 밀리므로 인덱스 계산 필요
    train_data = numbers_scaled[:1000]
    test_data = numbers_scaled[1000 - WINDOW_SIZE:] # 1001회를 맞추려면 앞데이터가 필요하니까 조금 겹치게 가져옴
    
    train_dataset = LottoDataset(train_data)
    test_dataset = LottoDataset(test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False) # 시계열이라 셔플 안 하는 게 보통
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # 모델 생성
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 사용하는 장치: {device}")
    
    model = LottoLSTM(input_size=6, hidden_size=HIDDEN_SIZE, num_layers=LAYERS, output_size=6).to(device)
    
    criterion = nn.MSELoss() # 손실함수 (정답과 예측값의 차이 계산)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- 학습 시작 ---
    print("🧠 학습을 시작합니다...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.6f}")

    # --- 모델 저장 ---
    torch.save(model.state_dict(), "lotto_lstm.pth")
    print("💾 모델 저장 완료! (lotto_lstm.pth)")
    
    # --- 검증 (1001회부터 예측 해보기) ---
    print("\n🔍 검증 데이터(1001회~) 예측 결과 확인")
    model.eval()
    with torch.no_grad():
        # 딱 하나만 예시로 테스트
        sample_x, sample_y = test_dataset[0] # 1001회차 예측을 위한 입력
        sample_x = sample_x.unsqueeze(0).to(device)
        
        prediction = model(sample_x)
        
        # 스케일링 된 걸 다시 원래 로또 번호로 복구
        predicted_numbers = scaler.inverse_transform(prediction.cpu().numpy())
        real_numbers = scaler.inverse_transform(sample_y.unsqueeze(0).numpy())
        
        print(f"예측값: {np.round(predicted_numbers).astype(int)}")
        print(f"정답값: {real_numbers.astype(int)}")