import pandas as pd
import matplotlib.pyplot as plt


url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
df = pd.read_csv(url)

print(df.head())  # 처음 5개 행 출력
print(df.info())  # 데이터 타입 & 결측치 확인
print(df.describe())  # 숫자형 데이터의 기초 통계량 출력


print(df.isnull().sum())  # 결측치 개수 확인

df["total_bedrooms"].fillna(df["total_bedrooms"].median(), inplace=True)
df.drop(columns=["ocean_proximity"], inplace=True)  # 쓸모없는 column 제거


# 가구당 방 개수, 가구당 침실 개수 추가
df["rooms_per_household"] = df["total_rooms"] / df["households"]
df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]

plt.figure(figsize=(8, 5))
plt.hist(df["median_house_value"], bins=50, color="skyblue", edgecolor="black")
plt.xlabel("Median House Value")
plt.ylabel("Frequency")
plt.title("Distribution of House Prices")
plt.show()


# 소득 vs 주택
plt.figure(figsize=(8, 5))
plt.scatter(df["median_income"], df["median_house_value"], alpha=0.3, color="blue")
plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.title("Income vs House Price")
plt.show()


# 위도 & 경도
plt.figure(figsize=(8, 5))
plt.scatter(df["longitude"], df["latitude"], c=df["median_house_value"], cmap="coolwarm", alpha=0.4)
plt.colorbar(label="Median House Value")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Geographical Distribution of House Prices")
plt.show()