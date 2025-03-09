from sklearn.datasets import fetch_openml
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim


# Boston Housing 데이터 가져오기
data = fetch_openml(name="boston", version=1, as_frame=True)
df = data.frame

# 데이터 확인
print(df.head())


# 결측치 확인
print(df.isnull().sum())

# 결측치 제거
df = df.dropna()


X = df.drop(columns=['MEDV'])  # 'MEDV'가 집값(타겟)
y = df['MEDV']


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)  # 0~1 범위로 변환


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Tensor 변환
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

"""
1️⃣ numpy나 pandas 데이터를 PyTorch Tensor로 변환
→ torch.tensor(X_train, dtype=torch.float32)
→ dtype=torch.float32는 PyTorch가 연산을 잘 수행하도록 실수형으로 변환

2️⃣ 타겟 데이터(y_train, y_test) 변환 시 .values 사용
→ pandas.Series를 Numpy 배열로 변환해서 torch.tensor()에 넣음

3️⃣ view(-1, 1) → 차원 변경 (reshape)

(-1, 1)은 1열짜리 2D 텐서로 변환
예를 들어 y_train이 [5, 10, 15] 같은 형태라면
view(-1, 1)을 적용하면 [[5], [10], [15]]로 변환
📌 이 과정이 필요한 이유
✅ PyTorch 모델에 데이터를 입력 가능한 형태로 맞추기 위해
✅ 특히 y 값은 단일 값이라도 2D 형태 ([batch_size, 1])로 맞춰야 함

✔ 한 줄 요약
👉 pandas/numpy 데이터를 PyTorch Tensor로 변환하고,
👉 y값을 2D로 reshape해서 학습 가능하게 만드는 과정! 🚀
"""



# 모델 정의
class RegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(RegressionModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layer(x)

"""
✔ nn.Module을 상속하여 모델 정의
✔ __init__()에서 레이어 정의 (nn.Linear, nn.ReLU)
✔ forward()에서 입력 데이터가 어떻게 흐르는지 지정
✔ nn.Sequential()로 간단하게 레이어 구성 가능

📌 결론:
nn.Linear() → 선형 변환 (Wx + b)
nn.ReLU() → 활성화 함수 적용 (비선형성 추가)
✔ 이렇게 층을 쌓아 학습 가능한 모델을 만든다! 🚀

"""


# 모델 생성
model = RegressionModel(input_dim=X_train.shape[1])

# 손실 함수 & 옵티마이저
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


# 학습
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    y_pred = model(X_train_tensor)
    loss = loss_fn(y_pred, y_train_tensor)

    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 평가
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test_tensor)
    test_loss = loss_fn(y_pred_test, y_test_tensor)
    print(f"Test Loss: {test_loss.item():.4f}")
