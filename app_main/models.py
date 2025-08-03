# models.py
from mongoengine import Document, fields
from datetime import datetime
from enum import Enum

class Status(Enum):
    NORMAL = "Normal"
    VIOLENCE = "Violence"
    WEAPONIZED = "Weaponized"

# 영상 모델 정의 (이미지도 저장할 수 있도록)
class Video(Document):
    video = fields.FileField()  # 영상 또는 이미지 파일 저장
    created_at = fields.DateTimeField(default=datetime.utcnow)  # 영상 업로드 시간
    status = fields.StringField(choices=[(status.name, status.value) for status in Status], default=Status.NORMAL.name)  # 상태 (Enum으로 처리)

    def __str__(self):
        return f"Video uploaded at {self.created_at}, Status: {self.status}"
