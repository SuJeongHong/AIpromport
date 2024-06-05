from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
import os
import json
import math
from sklearn.metrics.pairwise import cosine_similarity

# 직접 API 키 설정
api_key = "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"

# JSON 파일 경로
json_file_path = "korea_rest_update.json"
vector_data_file_path = "Untitled.json"

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

def get_nearest_places(user_location, places, n):
    places_with_distance = [(place["id"], calculate_distance(user_location, place["location"])) for place in places]
    sorted_places = sorted(places_with_distance, key=lambda x: x[1])
    return [place for place, distance in sorted_places[:n]]

def calculate_distance(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371  # 지구의 반지름 (단위: km)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lat2 - lon1)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance

with open(json_file_path, 'r', encoding='utf-8') as file:
    json_places = json.load(file)

with open(vector_data_file_path, 'rt', encoding='utf-8') as file:
    vector_data = json.load(file)

embeddings = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")

def embed_user_input(user_input):
    embedding = embeddings.embed_query(user_input)
    return embedding

def retrieve_from_vector_database(user_input, user_location, postcode):
    # postcode의 앞 3자리 추출
    postcode3 = postcode[:3]
    print("리트리버 우편 번호 앞 3자리", postcode3)
    filtered_places = [place for place in json_places if place["postcode"].startswith(postcode3)]
    nearest_place_ids_int = get_nearest_places(user_location, filtered_places, n=13)
    print(nearest_place_ids_int)
    nearest_place_ids = [str(item) for item in nearest_place_ids_int]
    vector_data_ids = list(vector_data.keys())
    small_db = {place_id: vector_data[place_id]["embedding"] for place_id in nearest_place_ids if place_id in vector_data_ids}

    # 사용자 입력에 가장 가까운 장소 찾기
    k = 5
    user_embedding = embed_user_input(user_input)
    top_places = [None] * k
    top_similarities = [0] * k
    most_similar_places = []

    for place_id, value in small_db.items():
        if value is not None:
            similarity = cosine_similarity([user_embedding], [value])[0][0]

            if similarity > min(top_similarities):
                min_similarity_index = top_similarities.index(min(top_similarities))
                top_places[min_similarity_index] = place_id
                top_similarities[min_similarity_index] = similarity

    for place_id in top_places:
        if place_id:
            similar_place = next((p for p in json_places if str(p["id"]) == str(place_id)), None)
            if similar_place:
                most_similar_places.append(similar_place)

    context = ""
    if most_similar_places:
        for i, place in enumerate(most_similar_places, 1):
            name = place["place_name"]
            type = place["category_name"]
            address = place["road_address_name"]
            rating = place["rating"]
            details = ", ".join(place["detail"])
            phone = place["phone"]
            context += f'''장소 이름은 '{name}'고 장소 타입은 '{type}'이다. 
    데이터베이스 내의 장소 중 현재 위치와 {i}번째로 가까운 장소이며, 주소는 '{address}'이고 평점은 '{rating}'점이다. 
    세부사항으로는 '{details}' 등이 있다.\n'''
    else:
        return "해당 장소 정보를 찾을 수 없습니다."

    context += f"\n사용자 입력: {user_input}"
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
