import os
from dotenv import load_dotenv
from mongoengine import connect

# .env 파일 로드
load_dotenv()

# 프로젝트 기본 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SECRET_KEY = os.getenv('DJANGO_SECRET_KEY', 'your_default_secret_key')
DEBUG = os.getenv('DEBUG', 'True') == 'True'
ALLOWED_HOSTS = []

# MongoDB Atlas 연결 URI
MONGO_URI = os.getenv('MONGO_URI')

# MongoDB 클라이언트 연결
connect(host=MONGO_URI)  # mongoengine의 connect() 함수 사용

# Django 기본 설정
INSTALLED_APPS = [
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',  # Static files 앱은 기본적으로 필요
    'app_main',  # MongoDB 연동을 위한 앱
]

# MIDDLEWARE 설정 (기본적인 보안 미들웨어와 인증 미들웨어만 포함)
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
]

ROOT_URLCONF = 'config.urls'
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Static files (CSS, JavaScript, Images)
STATIC_URL = '/static/'