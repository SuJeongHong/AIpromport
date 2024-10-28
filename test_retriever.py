from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
import os
import math
import requests
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity

# API 키 설정
api_key = "sk-proj-pyCLfEK2A4Ju3Jx7eS1VT3BlbkFJxM4iGvvDjEeZ78gZerP4"
# MongoDB key
connection_string = "mongodb+srv://hae081128:1213@test1.fe6gacs.mongodb.net/"

# MongoDB 서버에 연결
client = MongoClient(connection_string)
db = client['korea_rest']
cc_r = db['total_rest']  # 장소 데이터
cc_v = db['total_vector']  # 벡터 데이터

def get_address_from_coordinates(lat, lon):
    api_key = '61d4dab14a5504fb5a190597a427da0d'
    headers = {'Authorization': f'KakaoAK {api_key}'}
    url = 'https://dapi.kakao.com/v2/local/geo/coord2address.json'

    response = requests.get(url, headers=headers, params={
        'x': lon,
        'y': lat,
        'input_coord': 'WGS84'
    })

    result = response.json()
    if 'documents' in result and result['documents']:
        address_info = result['documents'][0].get('address')
        road_address_info = result['documents'][0].get('road_address')

        if road_address_info:
            postcode = road_address_info.get('zone_no', 'No postcode available')
        elif address_info:
            postcode = address_info.get('zip_code', 'No postcode available')
        else:
            postcode = 'No postcode available'

        address_name = address_info.get('address_name', 'No address available') if address_info else 'No address available'
        return address_name, postcode

    return None, "No address found"

# knn 함수: 사용자의 위치에서 가까운 n개의 위치
def get_nearest_places(user_location, places, n):
    places_with_distance = [(place["id"], calculate_distance(user_location, place["location"])) for place in places]
    sorted_places = sorted(places_with_distance, key=lambda x: x[1])
    return [place for place, distance in sorted_places[:n]]

# 거리 계산 함수
def calculate_distance(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371  # 지구의 반지름 (단위: km)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    return distance

embeddings = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")

def embed_user_input(user_input):
    return embeddings.embed_query(user_input)

def retrieve_from_vector_database(user_input, user_location, postcode):
    # postcode의 앞 3자리 추출
    postcode3 = postcode[:3]
    print("리트리버 우편 번호 앞 3자리", postcode3)

    # 1차 필터링: postcode가 일치하는 장소
    filtered_places = list(cc_r.find({"postcode": {"$regex": f"^{postcode3}"}}))
    print("1차:",filtered_places)
    # 필터링된 장소가 없으면 함수 종료
    if not filtered_places:
        return "해당 우편번호에 일치하는 장소가 없습니다."

    #2차 필터링: 사용자의 위치에서 가까운 n개 장소
    nearest_place_ids_int = get_nearest_places(user_location, filtered_places, n=50)
    nearest_place_ids = [str(item) for item in nearest_place_ids_int]

    # nearest_place_ids가 비어 있으면 함수 종료
    if not nearest_place_ids:
        return "사용자의 위치에서 가까운 장소를 찾을 수 없습니다."

    # total_vector의 문서 구조에 맞춰 id 필드를 사용하여 조회
    vector_docs = cc_v.find({"$or": [{str(nearest_id): {"$exists": True}} for nearest_id in nearest_place_ids]}, {'_id': 0})
    small_db_MongoDB = {str(nearest_id): doc[str(nearest_id)]["embedding"] for doc in vector_docs for nearest_id in nearest_place_ids if str(nearest_id) in doc}

    # 사용자 입력에 가장 가까운 장소 찾기
    k = 10
    user_embedding = embed_user_input(user_input)
    top_places = [None] * (k + 1)
    top_similarities = [float("-inf")] * (k + 1)
    most_similar_places = []

    for place_id, value in small_db_MongoDB.items():
        if value is not None:
            similarity = cosine_similarity([user_embedding], [value])[0][0]
            if similarity > min(top_similarities):
                min_similarity_index = top_similarities.index(min(top_similarities))
                top_places[min_similarity_index] = place_id
                top_similarities[min_similarity_index] = similarity

    for place_id in top_places:
        if place_id:
            similar_place = cc_r.find_one({"id": place_id})
            if similar_place:
                most_similar_places.append(similar_place)

    context = ""
    if most_similar_places:
        for rank, place in enumerate(most_similar_places, start=1):
            name = place.get("place_name", "Unknown")
            category = place.get("category_name", "Unknown")
            address = place.get("road_address_name", "Unknown")
            rating = place.get("rating", "Unknown")
            details = ", ".join(place.get("detail", []))
            context += f"k={rank}: 장소 이름은 '{name}'고 장소 타입은 '{category}'이다. 주소는 '{address}'이며 평점은 '{rating}'점이다. 세부사항으로는 '{details}' 등이 있다.\n"

    else:
        return "해당 장소 정보를 찾을 수 없습니다."

    context += f"\n사용자 입력: {user_input}"
    print(context)
    return context

# Langchain 설정
template = """Answer the question based only on the following context:

{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI(temperature=0, api_key=api_key, model_name="gpt-3.5-turbo")

def retrieve_with_location(input_dict):
    user_input = input_dict["question"]
    user_location = input_dict["location"]
    postcode = input_dict["postcode"]
    return retrieve_from_vector_database(user_input, user_location, postcode)

chain = (
    {"context": retrieve_with_location, "question": RunnablePassthrough(), "postcode": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

def responsetext(user_input, user_location, postcode):
    result = chain.invoke({"question": user_input, "location": user_location, "postcode": postcode})
    return result