# abnormalevents/urls.py

from django.urls import path
from . import views

urlpatterns = [
    # API 엔드포인트 경로를 명확하게 지정
    # 예: /api/events/abnormal/ 또는 /api/abnormal-events/ 등
    path('events/abnormal/', views.get_abnormal_events_api, name='get_abnormal_events_api'),
]
