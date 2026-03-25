import json
from typing import Any

import redis


class RedisFeatureStoreV2:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        decode_responses: bool = True,
    ):
        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=decode_responses,
        )

    def ping(self) -> bool:
        return bool(self.client.ping())

    def _set_json(self, key: str, payload: dict, ttl_seconds: int | None = None):
        value = json.dumps(payload, default=str)
        if ttl_seconds is not None:
            self.client.setex(key, ttl_seconds, value)
        else:
            self.client.set(key, value)

    def _get_json(self, key: str) -> dict:
        value = self.client.get(key)
        if value is None:
            return {}
        try:
            return json.loads(value)
        except Exception:
            return {}

    def put_user_features(self, user_id: str, features: dict, ttl_seconds: int | None = None):
        self._set_json(f"user:{user_id}", features, ttl_seconds)

    def get_user_features(self, user_id: str) -> dict:
        return self._get_json(f"user:{user_id}")

    def put_item_features(self, item_id: str, features: dict, ttl_seconds: int | None = None):
        self._set_json(f"item:{item_id}", features, ttl_seconds)

    def get_item_features(self, item_id: str) -> dict:
        return self._get_json(f"item:{item_id}")

    def put_metadata(self, key: str, payload: dict, ttl_seconds: int | None = None):
        self._set_json(f"meta:{key}", payload, ttl_seconds)

    def get_metadata(self, key: str) -> dict:
        return self._get_json(f"meta:{key}")


if __name__ == "__main__":
    store = RedisFeatureStoreV2()
    print("Redis reachable:", store.ping())

    store.put_user_features("demo_user", {"user_ctr_prior": 0.42, "user_prev_clicks": 12})
    store.put_item_features("demo_item", {"item_ctr_prior": 0.18, "item_prev_clicks": 9})

    print("User:", store.get_user_features("demo_user"))
    print("Item:", store.get_item_features("demo_item"))