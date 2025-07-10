from mongoengine import Document, fields
from enum import Enum
from datetime import datetime
import gridfs
from mongoengine.connection import get_db


# 상태를 나타내는 Enum 클래스 정의
class Status(Enum):
    NORMAL = "Normal"
    VIOLENCE = "Violence"
    WEAPONIZED = "Weaponized"


# 영상 모델 정의
class Video(Document):
    video = fields.FileField()  # 영상 파일 저장
    created_at = fields.DateTimeField(default=datetime.utcnow)  # 영상 업로드 시간
    status = fields.StringField(choices=[(status.name, status.value) for status in Status],
                                default=Status.NORMAL.name)  # 상태 (Enum으로 처리)

    def __str__(self):
        return f"Video uploaded at {self.created_at}, Status: {self.status}"

