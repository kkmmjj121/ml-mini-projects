from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time

# Chrome 옵션 설정
chrome_options = Options()
chrome_options.add_argument('--disable-gpu')  # GPU 가속 비활성화
chrome_options.add_argument('--headless')    # UI 없이 실행 (선택사항)

# Selenium을 사용하여 브라우저 열기
driver = webdriver.Chrome(options=chrome_options)

# YouTube 동영상 URL로 이동
video_url = ''  # url 입력
driver.get(video_url)

# 페이지 로딩을 위한 대기 시간 (필요에 따라 조정)
time.sleep(3)

# 댓글이 충분히 로드되도록 여러 번 스크롤
for _ in range(5):  # 스크롤을 5번 내리기 (이 숫자는 페이지 로딩 상황에 맞게 조정)
    driver.execute_script('window.scrollTo(0, document.documentElement.scrollHeight);')
    time.sleep(3)  # 스크롤 후 약간의 대기시간을 두어 댓글이 로드되도록 함

# 댓글 영역 가져오기
comments_section = driver.find_elements(By.CLASS_NAME, "style-scope ytd-comments")

# 댓글 내용과 좋아요 수를 추출하여 리스트에 저장
comments_with_likes = []

for section in comments_section:
    # 각 댓글을 찾기
    comment_models = section.find_elements(By.CLASS_NAME, "style-scope ytd-comment-view-model")
    
    for comment in comment_models:
        try:
            # 댓글 텍스트 추출
            comment_text = comment.find_element(By.CLASS_NAME, "yt-core-attributed-string--white-space-pre-wrap").text.strip()
            
            # 좋아요 수 추출
            like_button = comment.find_element(By.ID, "vote-count-middle")
            like_count_text = like_button.text.strip()
            like_count = 0

            # "천", "만"을 처리
            if '천' in like_count_text:
                like_count = float(like_count_text.replace('천', '').strip()) * 1000
            elif '만' in like_count_text:
                like_count = float(like_count_text.replace('만', '').strip()) * 10000
            else:
                try:
                    like_count = int(like_count_text)  # 그냥 숫자 처리
                except ValueError:
                    pass  # 숫자가 아니라면 0으로 처리

            comments_with_likes.append((comment_text, like_count))
        
        except Exception as e:
            continue  # 오류가 나면 무시하고 넘어감

# 좋아요 수를 기준으로 내림차순 정렬
comments_with_likes.sort(key=lambda x: x[1], reverse=True)

# 상위 10개 댓글 출력
for i, (comment, likes) in enumerate(comments_with_likes[:10]):
    print(f"{i+1}. {comment} - Likes: {likes}")

# 브라우저 종료
driver.quit()
