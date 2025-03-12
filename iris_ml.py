import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report


# 데이터 불러오기
iris = datasets.load_iris()

# DataFrame 변환
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target  # 타겟 값 추가
df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})  # 숫자를 품종 이름으로 변경

# 데이터 확인
print(df.head())
print(df.info())

# 결측치 확인
print(df.isnull().sum())

# 각 feature 별 분포 확인
sns.pairplot(df, hue='species')
plt.show()

# 상관 행렬을 히트맵으로 시각화
corr = df.drop(['species', 'target'], axis=1).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()

# Features와 Target 변수 분리
X = df.drop(['species', 'target'], axis=1)
y = df['target']

# 데이터 분할 (훈련/테스트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 정의 및 훈련
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 정확도 평가
accuracy = accuracy_score(y_test, y_pred)
print(f'로지스틱 회귀 모델 정확도: {accuracy * 100:.2f}%')




# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(cm)

# Classification Report
cr = classification_report(y_test, y_pred)
print('Classification Report:')
print(cr)