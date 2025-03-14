from dotenv import load_dotenv
import os
import requests
import sys
sys.stdout.reconfigure(encoding="utf-8")  # 출력 인코딩 변경

# .env 파일 로드

load_dotenv()

# 환경 변수에서 API 키 가져오기
api_key = os.getenv("YOUTUBE_API_KEY")

print(api_key)

video_url = ""  # 동영상 url 입력


def extract_video_id(url):
    # URL에서 '?v=' 뒤의 값만 추출
    if '?v=' in url:
        return url.split('?v=')[1].split('&')[0]  # ?v= 뒤의 값만 추출
    return None

# 동영상 ID 추출
video_id = extract_video_id(video_url)

# API 요청 URL (영상 정보 및 통계 데이터 가져오기)
url = f'https://www.googleapis.com/youtube/v3/videos?part=snippet,statistics&id={video_id}&key={api_key}'

response = requests.get(url)
video_data = response.json()

if 'items' in video_data:
    snippet = video_data['items'][0]['snippet']
    statistics = video_data['items'][0]['statistics']

    # 데이터 가져오기
    title = snippet.get('title', '제목 없음')
    like_count = int(statistics.get('likeCount', 0))
    dislike_count = int(statistics.get('dislikeCount', 0))  # 싫어요 수는 API에서 기본적으로 제공되지 않음
    view_count = int(statistics.get('viewCount', 0))
    comment_count = int(statistics.get('commentCount', 0))

    # 좋아요 비율 계산
    total_votes = like_count + dislike_count
    like_percentage = (like_count / total_votes * 100) if total_votes > 0 else 0

    # 결과 출력
    print(f"🎬 영상 제목: {title}")
    print(f"👍 좋아요 수: {like_count}")
    print(f"👎 싫어요 수: {dislike_count} (API에서 기본 제공 안함)")
    print(f"📊 좋아요 비율: {like_percentage:.2f}% (싫어요가 0이라 무조건 100퍼)")
    print(f"👀 조회수: {view_count}")
    print(f"💬 댓글 수: {comment_count}")

else:
    print("동영상 정보를 가져올 수 없습니다.")


print('\n\n\n')

url = f'https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId={video_id}&maxResults=100&order=relevance&key={api_key}'

response = requests.get(url)
comments_data = response.json()

comment_box = []


if 'items' in comments_data:
    for item in comments_data['items']:
        comment = item['snippet']['topLevelComment']['snippet']
        nickname = comment.get('authorDisplayName', '이름없음')
        text = comment.get('textDisplay', '댓글없음')
        comment_likes = comment.get('likeCount', 0)
        comment_box.append((nickname, text, comment_likes))

# 좋아요 순으로 정렬
comment_box.sort(key=lambda x: x[2], reverse=True)


print(f"{'닉네임':<15}{'댓글':<50}{'좋아요'}")
print("-" * 80)

for i in range(min(10, len(comment_box))):  # 10개보다 적을 수 있기 때문에 len()으로 안전하게 처리
    n = comment_box[i][0]  # 닉네임
    t = comment_box[i][1]  # 댓글
    c = comment_box[i][2]  # 좋아요 수
    
    # 댓글이 50자를 넘으면 생략하고 '...' 추가
    if len(t) > 50:
        t = t[:30] + '...'
    
    # 닉네임이 15자를 넘으면 생략하고 '...' 추가
    if len(n) > 15:
        n = n[:15] + '...'
    
    # 여기서 좋아요 부분을 맨 뒤에 고정시킴
    print(f"{n:<15}{t:<30}{c:>5}")




# https://developers.google.com/youtube/v3/docs/videos?hl=ko 
# https://developers.google.com/youtube/v3/docs/videos/list?hl=ko 

