import requests
from bs4 import BeautifulSoup

import sys
sys.stdout.reconfigure(encoding="utf-8")  # 출력 인코딩 변경

# 크롤링할 URL
url = "https://news.naver.com/main/ranking/popularDay.naver"

# 웹 페이지 요청
headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(url, headers=headers)
html = response.text

# HTML 파싱
soup = BeautifulSoup(html, "html.parser")

# 뉴스 제목 가져오기
news_titles = soup.select(".list_title")  # 클래스 이름이 'list_title'인 태그 선택 (네이버 뉴스 구조에 따라 변경 가능)

# 출력
for idx, title in enumerate(news_titles[:10]):  # 상위 10개 뉴스만 출력
    print(f"{idx + 1}. {title.text.strip()}")