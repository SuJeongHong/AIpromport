#localdb, 1차db
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
import os
import json
import math
from sklearn.metrics.pairwise import cosine_similarity

json_file_path = 'korea_rest_update.json'
vector_data_file_path ='vector_korea_update.json'

os.environ['OPENAI_API_KEY'] = "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"

# 사용자 위치 좌표 : 용산구청 / postcode 3자리 : 043
user_location = (37.5323013329348,126.99107111942)

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
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    return distance


with open(json_file_path, 'r', encoding='utf-8') as file:
    json_places = json.load(file)

with open(vector_data_file_path, 'rt', encoding='utf-8') as file:
    vector_data = json.load(file)

postcode3="043" # 사용자 위치 -> postcode로 변환한 3자리 값
filtered_places = [place for place in json_places if place["postcode"].startswith(postcode3)]

nearest_place_ids_int = get_nearest_places(user_location, filtered_places, n=13)
print(nearest_place_ids_int)
nearest_place_ids = [str(item) for item in nearest_place_ids_int]

vector_data_ids = list(vector_data.keys())

small_db={}
for i in range(len(nearest_place_ids)):
    if nearest_place_ids[i] in vector_data_ids:
        small_db[nearest_place_ids[i]] = vector_data[nearest_place_ids[i]]["embedding"]

embeddings = OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'], model="text-embedding-3-small")

def embed_user_input(user_input):
    embedding = embeddings.embed_query(user_input)
    return embedding

def retrieve_from_vector_database(user_input):
    # 사용자 입력에 가장 가까운 장소 찾기
    k=5
    user_embedding = embed_user_input(user_input)
    top_places = [None] * (k + 1)
    top_similarities = [0] * (k + 1)
    most_similar_places = []
    
    for place_id, value in small_db.items():
        if value is not None:
            similarity = cosine_similarity([user_embedding], [value])[0][0]

            # k개 반환
            if similarity > min(top_similarities):
                min_similarity_index = top_similarities.index(min(top_similarities))
                if len(top_similarities) < k:
                    top_places.append(place_id)
                    top_similarities.append(similarity)
                else:
                    top_places[min_similarity_index] = place_id
                    top_similarities[min_similarity_index] = similarity
            # 2개 반환
            # if similarity > min(top_similarities):
            #     index_to_replace = top_similarities.index(min(top_similarities))
            #     top_places[index_to_replace] = place_id
            #     top_similarities[index_to_replace] = similarity

    for place_id in top_places:
        similar_place = next((p for p in json_places if str(p["id"]) == str(place_id)), None)
        most_similar_places.append(similar_place)
    
    context = ""
    if most_similar_places:
        i = 0
        for place in most_similar_places[:-1]:
            i += 1
            name = place["place_name"]
            type = place["category_name"]
            address = place["road_address_name"]
            rating = place["rating"]
            details = ", ".join(place["detail"])
            phone = place["phone"]
            # 사용자 입력과 관련된 문맥 생성
            context += f'''장소 이름은 '{name}'고 장소 타입은 '{type}'이다. 
    데이터베이스 내의 장소 중 현재 위치와 {i}번째로 가까운 장소이며, 주소는 '{address}'이고 평점은 '{rating}'점이다. 
    세부사항으로는 '{details}' 등이 있다.\n'''
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
model = ChatOpenAI(temperature=0, api_key=os.environ['OPENAI_API_KEY'], model_name="gpt-3.5-turbo")

chain = (
    {"context": retrieve_from_vector_database , "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)


# user_input = input("사용자 입력을 입력하세요: ")
user_input = "소개팅하기 괜찮은 장소 추천해줘"
result = chain.invoke(user_input)
print("질문 >>", user_input)
print(result)
