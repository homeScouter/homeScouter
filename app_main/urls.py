# urls.py
from django.urls import path
from . import views

urlpatterns = [
    # 카메라 데이터 보내주는 api
    path('latest-frame/', views.get_latest_frame, name='latest_frame'),

    # 회원가입 api
    path('signup/', views.signup_api, name='signup'),

]
