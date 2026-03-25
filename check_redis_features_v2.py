from src.storage.redis_feature_store_v2 import RedisFeatureStoreV2


def main():
    store = RedisFeatureStoreV2()
    print("Redis reachable:", store.ping())

    meta = store.get_metadata("materialization_status")
    print("\nMaterialization metadata:")
    print(meta)

    # Try a couple of demo keys if present
    if meta.get("user_count"):
        print("\nRedis looks populated.")
    else:
        print("\nNo materialization metadata found yet.")


if __name__ == "__main__":
    main()