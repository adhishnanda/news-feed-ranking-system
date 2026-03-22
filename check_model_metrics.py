import json

with open("models_artifacts/metrics.json", "r", encoding="utf-8") as f:
    metrics = json.load(f)

print("=== Metrics Summary ===")
for model_name, model_metrics in metrics.items():
    if isinstance(model_metrics, dict):
        print(f"\n{model_name}")
        for k, v in model_metrics.items():
            print(f"  {k}: {v:.4f}")
    else:
        print(f"\n{model_name}: {model_metrics}")