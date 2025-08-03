# views.py
from django.http import HttpResponse, JsonResponse
from rest_framework.decorators import api_view
from camera_streamer.capture_api import capture_single_frame

@api_view(['GET'])
def get_latest_frame(request):
    """
    API 호출 시 RTSP에서 최신 프레임을 캡처하여 JPEG 이미지로 바로 반환.
    """
    jpeg_bytes = capture_single_frame()
    if jpeg_bytes is None:
        return JsonResponse({'error': 'Failed to capture frame'}, status=500)

    return HttpResponse(jpeg_bytes, content_type='image/jpeg')
