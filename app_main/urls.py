# app_main/urls.py

from django.urls import path
from . import views  # views에서 API 로직을 처리할 예정

urlpatterns = [
    path('example/', views.example_view, name='example'),  # 예시 엔드포인트
]