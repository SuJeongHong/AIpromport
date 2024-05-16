from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('answer.html')  # answer.html 파일을 루트 URL에서 렌더링

@app.route('/search', methods=['POST'])
def search():
    user_query = request.form['query']  # 사용자의 질문을 받아옴
    # 여기에 AI 호출 또는a 데이터 검색 로직을 추가
    response_text = process_query(user_query)  # 가상의 함수로, 실제 구현 필요
    return jsonify({'query': user_query, 'response': response_text})  # 결과를 JSON 형식으로 반환

def process_query(query):
    # 실제 AI 모델 호출 또는 검색 로직 구현
    # 임시 응답 예시
    return "This is a response to: " + query

if __name__ == '__main__':
    app.run(debug=True)


# from flask import Flask, request, jsonify, render_template
#
# app = Flask(__name__)
#
# # 리트리버 모듈 불러오기
# from test_retriever import retrieve_from_fake_database
#
#
# @app.route('/')
# def index():
#     return render_template('answer.html')  # answer.html 파일을 루트 URL에서 렌더링
#
#
# @app.route('/search', methods=['POST'])
# def search():
#     user_query = request.form['query']  # 사용자의 질문을 받아옴
#
#     # 리트리버 호출하여 사용자 질문 처리
#     response_text = retrieve_from_fake_database(user_query)
#
#     return jsonify({'query': user_query, 'response': response_text})  # 결과를 JSON 형식으로 반환
#
#
# if __name__ == '__main__':
#     app.run(debug=True)
