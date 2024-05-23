from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from test_retriever import responsetext

app = Flask(__name__)
CORS(app)  # 모든 도메인에 대해 CORS 허용

@app.route('/')
def index():
    return render_template('answer.html')  # answer.html 파일을 루트 URL에서 렌더링

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()  # 요청 데이터를 JSON 형식으로 파싱
    if data is None:
        return jsonify({'error': 'No data provided'}), 400

    user_query = data.get('query')  # 사용자의 질문을 받아옴
    user_location = data.get('location')  # 사용자의 위치를 받아옴

    if not user_query or not user_location:
        return jsonify({'error': 'Invalid data provided'}), 400

    # 리트리버를 통해 사용자의 질문 처리하고 응답 생성
    response_text = process_query(user_query, (user_location['lat'], user_location['lon']))

    # 응답을 JSON 형식으로 반환
    return jsonify({'query': user_query, 'response': response_text})

def process_query(query, user_location):
    # 리트리버 함수를 호출하여 사용자의 질문에 대한 정보를 가져옵니다.
    relevant_place_info = responsetext(query, user_location)

    # 리트리버에서 반환된 정보를 출력합니다.
    print("Retriever Response:", relevant_place_info)
    print("location : ", user_location)

    # 리트리버에서 가져온 정보를 응답으로 사용합니다.
    response_text = relevant_place_info

    return response_text

if __name__ == '__main__':
    app.run(debug=True)
