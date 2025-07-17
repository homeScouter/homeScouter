# camera_streamer/cache.py

import time

class MemoryCache:
    def __init__(self):
        self.cache = {}

    def set(self, key, value, timeout=10):
        """캐시에 데이터를 저장."""
        expiration_time = time.time() + timeout  # 만료 시간 설정
        self.cache[key] = {"value": value, "expires": expiration_time}

    def get(self, key):
        """캐시에서 데이터를 가져옴."""
        if key in self.cache:
            if self.cache[key]["expires"] > time.time():  # 만료되지 않았으면
                return self.cache[key]["value"]
            else:
                del self.cache[key]  # 만료된 데이터 삭제
        return None

    def clear(self):
        """전체 캐시 비우기."""
        self.cache.clear()

# 글로벌 캐시 객체
memory_cache = MemoryCache()
