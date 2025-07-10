# app_main/views.py

from django.http import JsonResponse
from .models import Photo, Status

def example_view(request):
    # 새로운 사진 데이터 삽입 예시 (이 부분은 실제 이미지 업로드와는 다르게 데이터만 삽입)
    new_photo = Photo(image=None, status=Status.NORMAL.name)  # 이미지 없이 테스트용 삽입
    new_photo.save()

    # MongoDB에서 저장된 데이터 조회
    photo = Photo.objects.first()

    return JsonResponse({
        'message': 'Hello, world!',
        'photo_created_at': photo.created_at,
        'photo_status': photo.status
    })
