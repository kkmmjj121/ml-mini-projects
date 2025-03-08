import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# root 데이터 경로
# train 학습용 테스트용 데이터셋 여부
# download=True 인터넷에서 다운
# transform 이미지 변환 - > torch에서도 쓸 수 있게
training_data = datasets.FashionMNIST(
    root= "data",
    train= True,
    download=True,
    transform= ToTensor()

)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# index로 데이터 접근
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

# plt 크기조절
figure = plt.figure(figsize=(8,8))


# 데이터 체크
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()



# DataLoader 설정

batch_size = 32

train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)





# 모델 정의
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()  # 이미지를 1차원 벡터로 변환
        self.fc1 = nn.Linear(28*28, 128)  # 첫 번째 완전연결층
        self.fc2 = nn.Linear(128, 64)  # 두 번째 완전연결층
        self.fc3 = nn.Linear(64, 10)  # 출력층 (10개 클래스)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 모델 생성
model = NeuralNetwork()


# 손실 함수
loss_fn = nn.CrossEntropyLoss()

# 옵티마이저 (SGD 사용)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# optimizer = optim.Adam(model.parameters(), lr=0.001)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # 모델을 GPU로 이동 (가능하면)

epochs = 5  # 학습 횟수

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")

    for batch, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # 예측
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            print(f"Batch {batch}, Loss: {loss.item():.4f}")

print("학습 완료!")


correct = 0
total = 0

model.eval()  # 평가 모드
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  # 가장 높은 확률의 클래스를 예측값으로 선택

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"테스트 정확도: {100 * correct / total:.2f}%")



"""

✅ CNN vs. nn.Linear (MLP)
CNN(합성곱 신경망)
→ 이미지 처리에 특화, 패턴(엣지, 텍스처 등)을 자동으로 추출
→ Conv2d(합성곱) + ReLU + MaxPool2d 사용
→ 예: 이미지 분류, 객체 탐지

nn.Linear (MLP, 완전연결 신경망)
→ 모든 뉴런이 서로 연결, 입력을 단순한 숫자로 변환
→ Linear + ReLU 사용
→ 예: 숫자 데이터, NLP, 간단한 이미지 분류

✅ 옵티마이저 역할
신경망이 더 좋은 가중치(weight)를 찾도록 업데이트
손실(loss)을 줄이는 방향으로 가중치 조정
예:
SGD (확률적 경사 하강법): 간단하지만 느림
Adam: 빠르고 효율적, 가장 많이 사용됨
✅ 순전파(Forward) vs. 역전파(Backward)
순전파 (Forward Pass)
→ 입력 데이터를 신경망을 통해 예측값 출력

역전파 (Backward Pass)
→ 예측값과 실제값의 차이를 기반으로 가중치를 조정 (오차 전파)
→ loss.backward()를 사용하여 기울기 계산

👉 순전파 → 결과 예측
👉 역전파 → 오차 수정 (학습 진행)

"""