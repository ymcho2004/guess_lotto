from fastapi import FastAPI, Query
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import copy
from fastapi.middleware.cors import CORSMiddleware

# --- 모델 구조 (변동 없음) ---
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
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

app = FastAPI()

origins = [
    "http://localhost:5173", # 리액트 개발 서버 주소
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 로드
device = torch.device('cpu')
model = LottoLSTM(input_size=6, hidden_size=128, num_layers=2, output_size=6)
try:
    model.load_state_dict(torch.load("lotto_lstm.pth", map_location=device))
except:
    print("모델 파일이 없어요!")
model.eval()

# 데이터 로드
df = pd.read_csv("lotto_history.csv")
scaler = MinMaxScaler()
scaler.fit_transform(df[['drwtNo1', 'drwtNo2', 'drwtNo3', 'drwtNo4', 'drwtNo5', 'drwtNo6']].values)

@app.get("/")
def read_root():
    return {"message": "🎰 로또 예측 AI (Deterministic Mode) 🎰"}

@app.get("/predict")
def predict_lotto(count: int = Query(5, ge=1, le=10)): 
    last_5_games = df.tail(5)[['drwtNo1', 'drwtNo2', 'drwtNo3', 'drwtNo4', 'drwtNo5', 'drwtNo6']].values
    input_data = scaler.transform(last_5_games)
    
    recommended_sets = []
    
    # 시드 고정 (새로고침해도 결과 유지)
    np.random.seed(42) 

    # --- AI가 끝까지 책임지는 함수 ---
    def get_ai_numbers(base_input, noise_level=0.0):
        # 1. 입력 데이터 준비 (노이즈 추가)
        noise = np.random.normal(0, noise_level, base_input.shape)
        noisy_input = base_input + noise
        input_tensor = torch.tensor(noisy_input, dtype=torch.float32).unsqueeze(0).to(device)
        
        # 2. 모델 예측
        with torch.no_grad():
            prediction = model(input_tensor)
        
        # 3. 결과 변환 (실수 -> 정수 리스트)
        pred_nums = scaler.inverse_transform(prediction.numpy())
        result = np.round(pred_nums).astype(int)[0]
        result = np.clip(result, 1, 45)
        return result.tolist() # 순수 파이썬 리스트로 반환

    # --- 사용자가 요청한 세트 수만큼 반복 ---
    for i in range(count):
        # 첫 번째 세트는 노이즈 없이(순수 실력), 그 뒤로는 노이즈 섞어서
        current_noise = 0.0 if i == 0 else 0.05
        
        # 1. 일단 AI한테 물어봄
        ai_picks = get_ai_numbers(input_data, noise_level=current_noise)
        
        # 2. 중복 제거
        unique_picks = sorted(list(set(ai_picks)))
        
        # 3. [핵심] 6개가 안 되면? AI한테 계속 다시 물어봐서 채움!
        attempts = 0
        while len(unique_picks) < 6:
            attempts += 1
            # "야, 다른 각도로 다시 생각해 봐" (노이즈를 조금씩 다르게 줌)
            # attempts가 늘어날수록 노이즈를 조금씩 키워서 새로운 숫자를 유도함
            retry_noise = current_noise + (attempts * 0.02)
            
            backup_picks = get_ai_numbers(input_data, noise_level=retry_noise)
            
            for num in backup_picks:
                if num not in unique_picks:
                    unique_picks.append(num)
                    if len(unique_picks) == 6:
                        break
            
            # (혹시나 무한루프 방지용 안전장치 - 100번 물어봐도 없으면 그때는 포기..하지만 그럴 일 없음)
            if attempts > 100:
                break
        
        recommended_sets.append(sorted(unique_picks))
    
    return {
        "count": count,
        "predictions": recommended_sets
    }