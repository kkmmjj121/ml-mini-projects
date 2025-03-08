import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 데이터 불러오기
california = fetch_california_housing()
df = pd.DataFrame(california.data, columns=california.feature_names)
df["Target"] = california.target  # 집값(Target) 추가

# 데이터 확인
print(df.head())  # 상위 5개 데이터 출력
print(df.info())  # 데이터 타입 확인
print(df.describe())  # 통계 요약

print(df.isnull().sum())

# 특성과 타겟 변수 분리
X = df.drop(columns=["Target"])  # 독립 변수 (특징 데이터)
y = df["Target"]  # 종속 변수 (집값)

# 훈련 데이터 & 테스트 데이터 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 정규화 (표준화)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 선형 회귀 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 예측 수행
y_pred = model.predict(X_test)

# 모델 평가 (평균제곱오차, MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

# 가중치(기울기)와 절편 출력
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)

# 실제 값 vs 예측 값 비교 그래프
plt.scatter(y_test, y_pred, alpha=0.5, color="blue")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Home Prices")
plt.show()


"""
1️⃣ 가중치(기울기)와 절편을 출력하는 이유
가중치 (Weights, Coefficients): 각 입력 변수(X)가 결과값(y)에 얼마나 영향을 미치는지를 나타냄
절편 (Intercept): 입력값이 모두 0일 때의 예측값
📌 출력하는 이유?

모델이 어떤 특징(변수)을 중요하게 생각하는지 확인 가능
예를 들어, 가중치가 크다면 그 변수는 집값에 큰 영향을 미친다는 의미!
2️⃣ MSE 말고도 회귀 평가 지표

///////from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

✔ MSE (Mean Squared Error, 평균제곱오차)

오차의 제곱을 평균낸 값 (낮을수록 좋음)
단위가 큼 → 해석이 어려울 수도 있음
✔ RMSE (Root Mean Squared Error, 평균제곱근오차)

MSE에 제곱근을 씌운 값
원래 값(y)과 같은 단위를 가짐 → 해석이 쉬움
RMSE = np.sqrt(MSE)
✔ MAE (Mean Absolute Error, 평균절대오차)

오차의 절댓값을 평균낸 것 (낮을수록 좋음)
MSE보다 이상치(Outlier)에 덜 민감
✔ R² Score (결정계수, 설명력)

0 ~ 1 사이 값 (1에 가까울수록 좋음)
모델이 데이터를 얼마나 잘 설명하는지 나타냄


✔ R² Score (결정계수) → 1에 가까울수록 좋음
모델이 데이터를 얼마나 잘 설명하는지 나타냄 (1이면 완벽한 예측)

✔ MSE, RMSE, MAE → 0에 가까울수록 좋음
값이 작을수록 예측 오차가 적다는 의미


3️⃣ 전처리 과정에서 중요한 점
✔ 결측치 처리 (Missing Values)

df.isnull().sum() 으로 확인 후 채우거나 삭제
숫자형 데이터 → 평균, 중앙값 대체 (df.fillna(df.mean()))
범주형 데이터 → 최빈값 대체
✔ 정규화 (Normalization) / 표준화 (Standardization)

데이터 크기 차이를 맞춰 모델 학습을 안정적으로 만듦
표준화: StandardScaler() (평균 0, 분산 1)
정규화: MinMaxScaler() (0~1 사이로 변환)
✔ 훈련 데이터 & 테스트 데이터 분리

데이터 과적합 방지 (train_test_split() 사용)
일반적으로 80:20 비율 사용
"""