# 위치 기반 맞춤형 추천 챗봇 (RAG 기반)

> 실시간 사용자 위치 정보를 바탕으로 장소를 추천하는 RAG 기반 챗봇 시스템입니다.  
> MongoDB 벡터 저장소, Langchain, OpenAI GPT, Kakao Map API 등을 활용하여 LLM의 한계를 보완하고, 개인화된 응답을 제공합니다.

## 기술 스택

- **Backend**: Flask, FastAPI, LangChain
- **LLM**: OpenAI GPT 3.5 Turbo (`gpt-3.5-turbo`)
- **임베딩**: OpenAI `text-embedding-3-small`, FAISS
- **DB**: MongoDB (장소 정보 및 벡터 저장)
- **API**: Kakao Map API (좌표 → 행정구역 변환)
- **ETC**: Python-dotenv, scikit-learn, cosine similarity

## 주요 기능

| 기능명                      | 설명 |
|---------------------------|------|
| 🧭 위치 기반 필터링        | 사용자의 현재 좌표를 바탕으로 우편번호 앞 3자리로 1차 필터링 |
| 📍 거리 기반 KNN 필터링    | 위경도 좌표 간 유클리디안 거리로 가장 가까운 장소 n개 추출 |
| 🧠 벡터 유사도 기반 매칭   | 사용자 질문과 장소 설명 임베딩 벡터 간 코사인 유사도 계산 |
| 💬 Langchain + GPT 응답    | 최종 장소 리스트 기반 자연어로 설명된 추천 응답 생성 |


## 프로젝트 구조

AIpromport/

├── sever.py # Flask 웹서버 (메인 실행 파일)

├── test_retriever.py # 장소 검색 및 GPT 응답 생성 (RAG 핵심 로직)

├── templates/

│ └── answer.html # 사용자 입력 UI (간단한 프론트엔드)

├── static/

│ └── styles.css # 선택적 스타일링

├── .env # API 키 저장 => .env파일 별도 생성 필요

└── requirements.txt # 설치 패키지 목록


## 실행 방법


```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. .env 파일 작성 (.env.example 참고)
KAKAO_MAP_KEY=your_kakao_api_key
OPENAI_API_KEY=your_openai_key
MONGO_DB_KEY=your_mongodb_uri

# 3. 서버 실행
python sever.py

접속 주소: http://127.0.0.1:5000/
```
### 기여 내용 
- 전체 RAG 파이프라인 기획 및 백엔드 설계
- Kakao Map API 좌표 → 행정구역 추출 처리 및 retriver연결 처
- Langchain 프롬프트 구성 및 응답 포맷 디자인
- KDBC 2024 한국데이터베이스학회 논문 공동 저자

## 📄 논문 정보

- **제목**: 위치 기반 맞춤형 추천을 위한 RAG 챗봇 시스템 개발  
- **학회**: KDBC 2024 한국데이터베이스학회  
- **저자**: 박시연, 이해영, 홍수정, 이기용  
- **PDF**: [논문 원문 다운로드](https://github.com/SuJeongHong/AIpromport/blob/main/KDBC_2024_paper_76_%EC%9C%84%EC%B9%98%EA%B8%B0%EB%B0%98%EB%A7%9E%EC%B6%A4%ED%98%95%EC%B6%94%EC%B2%9C%EC%9D%84%20%EC%9C%84%ED%95%9C%20RAG%EC%B1%97%EB%B4%87%EC%8B%9C%EC%8A%A4%ED%85%9C%EA%B0%9C%EB%B0%9C%20%EB%85%BC%EB%AC%B8_%ED%99%8D%EC%88%98%EC%A0%95.pdf)

