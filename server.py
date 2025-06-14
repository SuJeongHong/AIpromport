from flask import Flask, request, jsonify, render_template
from test_retriever import responsetext
import requests,os

from dotenv import load_dotenv

load_dotenv()  # .env 파일의 변수들을 불러온다

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('answer.html')  # answer.html 파일을 루트 URL에서 렌더링

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()  # JSON 데이터 받아오기
    user_query = data.get('query')
    user_location = data.get('location')

    if user_query and user_location:
        latitude = user_location.get('lat')
        longitude = user_location.get('lon')

        # 사용자 위치 좌표를 튜플로 만듦
        location = (33.430403, 126.927703)#(latitude, longitude)

        # 리트리버를 통해 사용자의 질문 처리하고 응답 생성
        response_text = process_query(user_query, location)

        # 응답을 JSON 형식으로 반환
        return jsonify({'query': user_query, 'response': response_text})
    else:
        return jsonify({'error': 'Invalid input'}), 400

def process_query(query, user_location):
    # 현재 위치 좌표 텍스트 변환
    address_name, postcode = get_address_from_coordinates(*user_location)
    if postcode == '' or postcode == 'No postcode available':
        # 주변 좌표를 기반으로 우편번호 앞 3자리 숫자를 얻음
        postcode = get_nearby_postcode(user_location)
        print("repostcode", postcode)
    else:
        print("우편번호: ", postcode)
    #print("location : ", user_location)
    #print("text장소 : ", address_name)

    # 리트리버 함수를 호출하여 사용자의 질문에 대한 정보를 가져옵니다.
    relevant_place_info = responsetext(query, user_location, postcode)

    # 리트리버에서 반환된 정보를 출력합니다.
    #print("Retriever Response:", relevant_place_info)

    # 리트리버에서 가져온 정보를 응답으로 사용합니다.
    response_text = relevant_place_info

    return response_text

# 현재 위치 좌표 텍스트와 우편번호로 변환
def get_address_from_coordinates(lat, lon):
    api_key = os.getenv("KAKAO_MAP_KEY")
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

def get_nearby_postcode(location):
    # 주변 좌표들을 생성하여 우편번호를 검색
    lat, lon = location
    nearby_locations = [
        (lat + 0.01, lon),
        (lat - 0.01, lon),
        (lat, lon + 0.01),
        (lat, lon - 0.01),
        (lat + 0.005, lon + 0.005),
        (lat - 0.005, lon - 0.005),
        (lat + 0.02, lon),
        (lat - 0.02, lon),
        (lat, lon + 0.02),
        (lat, lon - 0.02)
    ]

    for loc in nearby_locations:
        address_name, postcode = get_address_from_coordinates(*loc)
        if postcode != 'No postcode available':
            return postcode[:3]  # 우편번호 앞 3자리 반환

    return 'No postcode available'

if __name__ == '__main__':
    app.run(debug=True)
