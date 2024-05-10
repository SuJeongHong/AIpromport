import json
import os
from langchain_openai import OpenAIEmbeddings

# 환경 변수에서 API 키를 가져옴
os.environ['OPENAI_API_KEY'] = "YOUR_API_KEY"


def process_query(user_input):
    # 사용자의 입력을 받아 장소를 검색하고 결과를 반환하는 함수

    # JSON 파일 로드
    with open('restaurant.json', 'r', encoding='utf-8') as file:
        json_places = json.load(file)

    # 가까운 장소 찾기 로직 (위에서 설명한 로직을 여기에 구현)
    nearest_places = get_nearest_places((37.534274, 126.970481), json_places, 5)

    # 임베딩과 유사도 계산 로직
    embeddings = OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'], model="text-embedding-3-small")
    user_embedding = embeddings.embed_query(user_input)
    # 이어서 유사도 계산 로직 구현

    # 결과 생성 및 반환
    result = create_response_context(nearest_places)
    return result


def create_response_context(places):
    # 선택된 장소에 대한 정보를 문자열로 변환하여 반환
    context = ""
    for place in places:
        context += f"Name: {place['name']}, Address: {place['address']}, Rating: {place['rating']}\n"
    return context
