from pathlib import Path
import json


PATH = Path("data/gold/metrics_simulated_v3.json")


def main():
    if not PATH.exists():
        print(f"Missing file: {PATH}")
        return

    with open(PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("\n=== SIMULATED V3 METRICS ===")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
    