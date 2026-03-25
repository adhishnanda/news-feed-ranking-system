class OnlineFeatureStoreStub:
    def __init__(self):
        self.store = {}

    def put_features(self, key: str, values: dict):
        self.store[key] = values

    def get_features(self, key: str):
        return self.store.get(key, {})

    def get_user_features(self, user_id):
        return self.get_features(f"user:{user_id}")

    def get_item_features(self, item_id):
        return self.get_features(f"item:{item_id}")


if __name__ == "__main__":
    store = OnlineFeatureStoreStub()
    store.put_features("user:1", {"user_prev_clicks": 4, "user_ctr_prior": 0.3})
    store.put_features("item:101", {"item_prev_clicks": 11, "item_ctr_prior": 0.12})

    print(store.get_user_features(1))
    print(store.get_item_features(101))