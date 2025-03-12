from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import sys
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")  # 출력 인코딩 변경


chrome_options = Options()
chrome_options.add_argument("--headless")  # 브라우저 UI를 표시하지 않음
chrome_options.add_argument("--disable-gpu")  # GPU 사용 안함 (성능 향상)

# Chrome 브라우저 실행 (한 번만)
driver = webdriver.Chrome(options=chrome_options)

data = []
flag = 0

for i in range(1, 30):
    # 각 페이지 이동
    url = f"https://hearthstone.blizzard.com/ko-kr/community/leaderboards/?region=AP&leaderboardId=arena&seasonId=54&page={i}"
    driver.get(url)

    # 특정 클래스가 로드될 때까지 대기 (최대 10초)
    wait = WebDriverWait(driver, 10)
    table = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "LeaderboardsTable-Rendered")))

    # `LeaderboardsTable-Rendered` 내부에서 `row` 클래스를 가진 모든 요소 찾기
    rows = table.find_elements(By.CLASS_NAME, "row")

    # 각 row에서 데이터를 가져와 리스트로 저장
    for row in rows:
        columns = row.text.split("\n")  # 공백 기준으로 데이터 나누기
        if columns[1] == "Flurry":
            flag = 1
        data.append(columns)
    if flag:
        break

# 브라우저 종료 (한 번만)
driver.quit()

# DataFrame 생성 (컬럼명은 실제 데이터 확인 후 수정 가능)
df = pd.DataFrame(data, columns=["순위", "플레이어명", "평균 승수"])

# print(df.head())

flurry_data = df[df["플레이어명"] == "Flurry"]

# 결과 출력
if not flurry_data.empty:
    print(flurry_data)
    
else:
    print("Flurry 플레이어가 없습니다.")