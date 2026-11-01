import argparse
import pandas as pd
from src.fraud_pipeline import resolve_target, train_and_evaluate, save_artifacts

def main():
    parser = argparse.ArgumentParser(description="Train a financial fraud detection model.")
    parser.add_argument("--data", required=True, help="Path to transaction CSV")
    parser.add_argument("--target", default=None, help="Binary target column (0=non-fraud, 1=fraud)")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    target = resolve_target(df, args.target)
    print(f"Loaded {len(df):,} rows and {len(df.columns):,} columns")
    print(f"Target: {target}")
    print(df[target].value_counts(dropna=False))
    result = train_and_evaluate(df, target, args.random_state)
    save_artifacts(result)

    print(f"\nSelected model: {result['model_name']}")
    for name, values in result["metrics"].items():
        print(f"\n{name}")
        for metric in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
            print(f"  {metric}: {values[metric]:.4f}")
    print("\nClassification report:")
    print(result["classification_report"])
    print("\nModel and reports saved.")

if __name__ == "__main__":
    main()
