import os
from dotenv import load_dotenv

# Firebase Admin SDK를 위한 import
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore  # Firestore 클라이언트를 사용하기 위한 import
from firebase_admin import auth # <-- 추가: Firebase Authentication 모듈 import

# .env 파일 로드 (가장 상단에 위치)
load_dotenv()

# --- 기존 Django 프로젝트 기본 설정은 그대로 유지 ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SECRET_KEY = os.getenv('DJANGO_SECRET_KEY', 'your_default_secret_key_if_not_in_env')
DEBUG = os.getenv('DEBUG', 'True') == 'True'
ALLOWED_HOSTS = []

# --- Firebase Admin SDK 및 Firestore 초기화 시작 ---
# 앱이 이미 초기화되었는지 확인하여 중복 초기화를 방지합니다.
# Django는 여러 번 스크립트를 로드할 수 있으므로 중요합니다.
if not firebase_admin._apps:
    try:
        SERVICE_ACCOUNT_KEY_PATH = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if not SERVICE_ACCOUNT_KEY_PATH:
            raise RuntimeError("환경 변수 GOOGLE_APPLICATION_CREDENTIALS가 설정되어 있지 않습니다.")

        # 파일 존재 여부 확인 (절대경로여야 하며, 상대경로일 경우 Django 실행 위치 기준임 유의)
        if not os.path.isfile(SERVICE_ACCOUNT_KEY_PATH):
            raise FileNotFoundError(f"서비스 계정 키 파일을 찾을 수 없습니다: {SERVICE_ACCOUNT_KEY_PATH}")

        # --- 디버깅용: 서비스 계정 키 경로 확인 ---
        print(f"DEBUG: 확인된 서비스 계정 키 경로: {SERVICE_ACCOUNT_KEY_PATH}")

        cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)

        firebase_admin.initialize_app(cred, {
            'projectId': 'home-scouter-50835',  # 본인 Firebase 프로젝트 ID로 수정하세요
            # 만약 Realtime Database 사용 시 아래 주석 해제하고 URL 입력
            # 'databaseURL': 'https://home-scouter-50835.firebaseio.com'
        })
        print("Firebase Admin SDK 초기화 성공!")

    except Exception as e:
        print(f"Firebase Admin SDK 초기화 중 오류 발생: {e}")
        # 심각한 에러면 앱 시작 중단을 원할 경우 아래 주석 해제
        # raise
else:
    print("Firebase Admin SDK는 이미 초기화되어 있습니다.")

# Firestore 클라이언트 인스턴스 생성
# 이 'db' 객체를 사용하여 Firestore에 데이터를 읽고 쓸 것입니다.
# settings.py 밖에서 이 db 객체를 사용하려면 import해서 사용하면 됩니다.
# 예: from config.settings import db
db = firestore.client()
# --- Firebase Admin SDK 및 Firestore 초기화 끝 ---


# --- Django 기본 설정은 그대로 유지 ---
INSTALLED_APPS = [
    'corsheaders',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'app_main',
    'rest_framework',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
]

CORS_ALLOWED_ORIGINS = [
    "http://localhost:3333",
]

ROOT_URLCONF = 'config.urls'
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
STATIC_URL = '/static/'
