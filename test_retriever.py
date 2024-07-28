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
cc_r = db['500_seoul_rest']  # 장소 데이터
cc_v = db['500_seoul_vector']  # 벡터 데이터

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

def calculate_distance(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371  # 지구의 반지름 (단위: km)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance

embeddings = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")

def embed_user_input(user_input):
    embedding = embeddings.embed_query(user_input)
    return embedding

def retrieve_from_vector_database(user_input, user_location, postcode):
    # postcode의 앞 3자리 추출
    postcode3 = postcode[:3]
    print("리트리버 우편 번호 앞 3자리", postcode3)

    # 1차 db: postcode가 같은 장소 리스트 저장
    filtered_places = list(cc_r.find({"postcode": {"$regex": f"^{postcode3}"}}))
    print("1차 db: ", filtered_places)
    # 2차 db: 1차 db에서 사용자의 위치에서 가까운 n개 장소 저장
    nearest_place_ids_int = get_nearest_places(user_location, filtered_places, n=13)
    print("2차 db: ",nearest_place_ids_int)
    nearest_place_ids = [str(item) for item in nearest_place_ids_int]

    vector_data_ids = []
    cc_v_document = cc_v.find()
    for doc in cc_v_document:
        vector_data_ids.extend([key for key in doc.keys() if key.isdigit()])

    small_db = {}
    for num_id in nearest_place_ids:
        document = cc_v.find_one({num_id: {"$exists": True}}, {'_id': 0})
        if document:
            small_db[num_id] = document[num_id]["embedding"]
            print("smalldb: ",small_db)
        else:
            print(f"No document found for id {num_id}")

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
            similar_place = cc_r.find_one({"id": place_id})
            if similar_place:
                most_similar_places.append(similar_place)

    context = ""
    if most_similar_places:
        for i, place in enumerate(most_similar_places, 1):
            name = place.get("place_name", "Unknown")
            type = place.get("category_name", "Unknown")
            address = place.get("road_address_name", "Unknown")
            rating = place.get("rating", "Unknown")
            details = ", ".join(place.get("detail", []))
            phone = place.get("phone", "Unknown")
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