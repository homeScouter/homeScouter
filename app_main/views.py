from django.http import JsonResponse
from .models import Video, Status

def example_view(request):
    # 테스트용으로 영상 데이터 삽입 (파일 없이 데이터만 삽입)
    new_video = Video(video=None, status=Status.NORMAL.name)  # 영상 없이 테스트용 삽입
    new_video.save()

    # MongoDB에서 저장된 첫 번째 영상 데이터 조회
    video = Video.objects.first()

    return JsonResponse({
        'message': 'Hello, world!',
        'video_created_at': video.created_at,
        'video_status': video.status
    })
