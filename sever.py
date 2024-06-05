from flask import Flask, request, jsonify, render_template
from test_retriever import responsetext
import requests

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
        location = (latitude, longitude)

        # 리트리버를 통해 사용자의 질문 처리하고 응답 생성
        response_text, error_message = process_query(user_query, location)

        if error_message:
            return jsonify({'error': error_message}), 400

        # 응답을 JSON 형식으로 반환
        return jsonify({'query': user_query, 'response': response_text})
    else:
        return jsonify({'error': 'Invalid input'}), 400

def process_query(query, user_location):
    # 리트리버 함수를 호출하여 사용자의 질문에 대한 정보를 가져옵니다.
    relevant_place_info = responsetext(query, user_location)

    # 리트리버에서 반환된 정보를 출력합니다.
    print("Retriever Response:", relevant_place_info)
    print("location : ", user_location)

    # 현재 위치 좌표 텍스트 변환
    address_info, postcode, error_message = get_address_from_coordinates(*user_location)
    if error_message:
        return None, error_message

    print("위치: ", address_info)
    print("우편번호: ", postcode)

    # 리트리버에서 가져온 정보를 응답으로 사용합니다.
    response_text = f"{relevant_place_info}\nLocation: {address_info}\nPostcode: {postcode}"

    return response_text, None

# 현재 위치 좌표 텍스트와 우편번호로 변환
def get_address_from_coordinates(lat, lon):
    api_key = '61d4dab14a5504fb5a190597a427da0d'  # Kakao API키
    headers = {'Authorization': f'KakaoAK {api_key}'}
    url = 'https://dapi.kakao.com/v2/local/geo/coord2address.json'
    # 가까운 역지오 코딩 API URL변수에 저장

    response = requests.get(url, headers=headers, params={  # API에 GET요청 보냄
        'x': lon,
        'y': lat,
        'input_coord': 'WGS84'  # 좌표계 타입
    })

    result = response.json()  # 응답을 Json형태로 파싱
    if 'documents' in result and result['documents']:  # document키 비어있지 않은 경우 주소 정보 추출
        address_info = result['documents'][0]['address']  # address키에 해당하는 주소 정보 추출
        road_address_info = result['documents'][0].get('road_address')  # 'road_address' 키에 해당하는 도로명 주소 정보가 있을 경우 이를 추출

        address_name = address_info.get('address_name', 'No address available')  # 일반주소

        # 도로명 주소가 있는 경우 zip_code 사용
        if road_address_info:
            zip_code = road_address_info.get('zone_no', 'No postcode available')
        else:
            # 도로명 주소가 없는 경우 일반 주소의 zip_code 사용
            zip_code = address_info.get('zip_code', 'No postcode available')

        return address_name, zip_code, None

    return None, None, "No address found"

if __name__ == '__main__':
    app.run(debug=True)
