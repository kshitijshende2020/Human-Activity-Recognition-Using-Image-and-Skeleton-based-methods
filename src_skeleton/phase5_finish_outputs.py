# =====================================================================
# PHASE 5 - FINISHING CELL (run in the SAME Colab session, right after the crashed cell)
# Uses the predictions that are already in memory (preds_store, rep, y_te).
# NOTHING is re-trained and the test set is NOT evaluated again.
# =====================================================================
pd.concat({t: pd.DataFrame({k: v for k, v in rep[t].items() if isinstance(v, dict)}).T for t in rep},
          axis=1).to_csv(f"{LOG_DIR}/test_per_class_seed42.csv")

for t, title in (("A", "Model A (single pose)"), ("B", "Model B (pose sequence)")):
    cm = confusion_matrix(y_te, preds_store[t], labels=list(range(12)))
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(f"{LOG_DIR}/test_confusion_{t}.csv")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label"); ax.set_title(f"Confusion Matrix - Test Set - {title}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"); plt.tight_layout()
    fig.savefig(f"{LOG_DIR}/test_confusion_{t}.png", dpi=150, bbox_inches="tight"); plt.show()
    if t == "B":
        pairs = sorted([(cm[i, j], i, j) for i in range(12) for j in range(12) if i != j], reverse=True)[:8]
        print("Most common mistakes of Model B (true -> predicted : count)")
        for c, i, j in pairs: print(f"  {CLASS_NAMES[i]} -> {CLASS_NAMES[j]} : {c}")

with open(LOCK, "w") as f: f.write("Test set evaluated once. Do not re-run.")
print(f"\nSaved to {LOG_DIR}: test_results_per_seed.csv, test_per_class_seed42.csv, confusion matrices (.png/.csv)")
