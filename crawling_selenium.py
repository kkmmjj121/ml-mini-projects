from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time

# Chrome 옵션 설정
options = Options()
# options.add_argument('--headless')  # 브라우저 창을 띄우지 않고 실행 (필요 시 주석 처리)
options.add_argument('--disable-gpu')  # GPU 비활성화
options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')  # User-Agent 설정

# Chrome 브라우저 실행

driver = webdriver.Chrome(options=options)

# 구글 검색 페이지 열기
driver.get('https://www.google.com')

# 구글 검색창을 찾고 'Python' 검색어 입력
search_box = driver.find_element(By.NAME, 'q')
search_box.send_keys('Python')
search_box.submit()  # 검색 제출

# 페이지 로딩 대기
time.sleep(2)  # 검색 결과가 로딩될 때까지 잠시 대기

# 검색 결과에서 첫 번째 결과의 제목 추출
first_result = driver.find_element(By.XPATH, '(//h3)[1]')  # 첫 번째 <h3> 태그 (검색 결과 제목)
print('첫 번째 검색 결과 제목:', first_result.text)

# 브라우저 종료
driver.quit()


# 캡챠에 막히는걸 확인함