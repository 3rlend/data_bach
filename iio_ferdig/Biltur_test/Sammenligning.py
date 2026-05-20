import pandas as pd
import numpy as np
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import glob
import os

# ── Konfig ──────────────────────────────────────────────────────────────────
DATA_DIR = "data"
TOP_K    = 20
# ────────────────────────────────────────────────────────────────────────────

def les_filer(mappe, label, start_id=0):
    rader = []
    labels = {}
    win_id = start_id
    for fil in sorted(glob.glob(os.path.join(mappe, "*.csv"))):
        df = pd.read_csv(fil)[["x", "y", "z"]].dropna().reset_index(drop=True)
        df["id"] = win_id
        labels[win_id] = label
        rader.append(df)
        print(f"  {os.path.basename(fil)}: {len(df)} rader → id={win_id}")
        win_id += 1
    return pd.concat(rader, ignore_index=True), labels

print("Leser data...")
df_biltur, lab_biltur = les_filer(os.path.join(DATA_DIR, "biltur"), label=0, start_id=0)
df_fast,   lab_fast   = les_filer(os.path.join(DATA_DIR, "fast"),   label=1, start_id=len(lab_biltur))

alle   = pd.concat([df_biltur, df_fast], ignore_index=True)
labels = pd.Series({**lab_biltur, **lab_fast})

print(f"\n  Biltur-filer: {len(lab_biltur)}")
print(f"  Fast-filer  : {len(lab_fast)}")

print("\nEkstraherer tsfresh-features (EfficientFCParameters)...")
features = extract_features(
    alle[["id", "x", "y", "z"]],
    column_id="id",
    default_fc_parameters=EfficientFCParameters(),
    impute_function=None,
    disable_progressbar=False,
)
features = features.fillna(0).replace([np.inf, -np.inf], 0)

# ── Feature-seleksjon med F-score ────────────────────────────────────────────
print(f"\nVelger topp {TOP_K} features med F-score...")
selector  = SelectKBest(f_classif, k=TOP_K)
selector.fit(features, labels)

topp_idx  = selector.get_support(indices=True)
topp_df   = pd.DataFrame({
    "feature": features.columns[topp_idx],
    "F_score": selector.scores_[topp_idx]
}).sort_values("F_score", ascending=False).reset_index(drop=True)
topp_df.index += 1

print(f"\nTopp {TOP_K} features:\n")
print(topp_df.to_string())
topp_df.to_csv("topp_features.csv")

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(topp_df["feature"][::-1], topp_df["F_score"][::-1], color="steelblue")
ax.set_xlabel("F-score")
ax.set_title(f"Topp {TOP_K} tsfresh-features — Biltur vs Fast")
plt.tight_layout()
plt.savefig("topp_features.png", dpi=150)
print("\n✓ Plot lagret → topp_features.png")
plt.show()