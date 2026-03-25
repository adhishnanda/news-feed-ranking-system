from pathlib import Path
import json

PATH = Path("data/gold/metrics_simulated_v2.json")

def main():
    if not PATH.exists():
        print("Metrics file not found")
        return

    with open(PATH) as f:
        data = json.load(f)

    print("\n=== SIMULATED V2 METRICS ===")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()