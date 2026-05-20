import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — unngår Tkinter-støy
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import hashlib
import json
import re
import warnings
import argparse
import sys

# tsfresh bibloteker som fikser features
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    f1_score,
    roc_curve,
    auc,
    precision_recall_fscore_support
)
from sklearn.feature_selection import SelectKBest, f_classif

# --------------------------------------------------------------------------- #
# KONFIGURASJON                                                               #
# --------------------------------------------------------------------------- #
DATA_ROOT  = Path("./data")
CACHE_DIR  = Path("./cache")
MODEL_DIR  = Path("./models")
SAMPLING_HZ    = 833
AXES           = ["x", "y", "z"]
TOP_K_FEATURES = 100
RANDOM_STATE   = 42

# Standardverdier — kan overstyres med --window og --overlap fra kommandolinjen
WINDOW_SECONDS = 6.0
WINDOW_OVERLAP = 0.5


def _parse_cli_args():
    parser = argparse.ArgumentParser(description="Random Forest paa vibrasjonsdata")
    parser.add_argument("--window", type=float, default=WINDOW_SECONDS,
                        help=f"Vinduslengde i sekunder (default: {WINDOW_SECONDS})")
    parser.add_argument("--overlap", type=float, default=WINDOW_OVERLAP,
                        help=f"Overlapp mellom vinduer 0-1 (default: {WINDOW_OVERLAP})")
    args, _ = parser.parse_known_args()
    return args


_args = _parse_cli_args()
WINDOW_SECONDS = _args.window
WINDOW_OVERLAP = _args.overlap

WINDOW_SAMPLES = int(WINDOW_SECONDS * SAMPLING_HZ)
STEP_SAMPLES   = max(1, int(WINDOW_SAMPLES * (1 - WINDOW_OVERLAP)))

WINDOW_TAG = f"{WINDOW_SECONDS:g}s"
CACHE_DIR  = CACHE_DIR  / WINDOW_TAG
MODEL_DIR  = MODEL_DIR  / WINDOW_TAG
PLOT_DIR   = Path("./plots") / WINDOW_TAG
PLOT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[CONFIG] vindu = {WINDOW_SECONDS}s ({WINDOW_SAMPLES} samples), "
      f"overlapp = {WINDOW_OVERLAP}, steg = {STEP_SAMPLES} samples")
print(f"[CONFIG] cache   -> {CACHE_DIR}")
print(f"[CONFIG] modeller -> {MODEL_DIR}")
print(f"[CONFIG] plott   -> {PLOT_DIR}")

# --------------------------------------------------------------------------- #
# CACHE-HÅNDTERING                                                            #
# --------------------------------------------------------------------------- #
def _config_fingerprint() -> str:
    cfg = {
        "sampling_hz":    SAMPLING_HZ,
        "window_seconds": WINDOW_SECONDS,
        "window_overlap": WINDOW_OVERLAP,
        "axes":           AXES,
    }
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]

def _data_fingerprint(data_root: Path) -> str:
    csv_files   = sorted(data_root.rglob("*.csv"))
    fingerprint = "_".join(f"{p}:{p.stat().st_mtime:.0f}" for p in csv_files)
    return hashlib.md5(fingerprint.encode()).hexdigest()[:8]

def _cache_paths(data_root: Path):
    tag = f"{_config_fingerprint()}_{_data_fingerprint(data_root)}"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return (
        CACHE_DIR / f"windowed_{tag}.joblib",
        CACHE_DIR / f"features_{tag}.joblib",
    )

def load_cache(data_root: Path):
    win_path, feat_path = _cache_paths(data_root)
    if win_path.exists() and feat_path.exists():
        print(f"✓ Gyldig cache funnet ({win_path.name})")
        print("  Laster vinduer og features fra disk — hopper over tsfresh...")
        long_df, y, groups = joblib.load(win_path)
        X = joblib.load(feat_path)
        return long_df, y, groups, X
    for old in CACHE_DIR.glob("windowed_*.joblib"):
        old.unlink()
    for old in CACHE_DIR.glob("features_*.joblib"):
        old.unlink()
    return None

def save_cache(data_root: Path, long_df, y, groups, X):
    win_path, feat_path = _cache_paths(data_root)
    joblib.dump((long_df, y, groups), win_path)
    joblib.dump(X, feat_path)
    print(f"✓ Cache lagret → {CACHE_DIR}/")

# --------------------------------------------------------------------------- #
# 1. INNLESING + VINDUSPLITTING                                               #
# --------------------------------------------------------------------------- #
def load_and_window(data_root: Path):
    rows   = []
    labels = {}
    groups = {}
    sample_id = 0

    csv_files = sorted(list(data_root.rglob("*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"Fant ingen CSV-filer i {data_root.resolve()}")

    print(f"Prosesserer filer med {WINDOW_SECONDS}s vinduer...")

    skipped_idle = 0
    skipped_short = 0
    processed = 0

    for csv_path in csv_files:
        rel_parts = csv_path.relative_to(data_root).parts
        if len(rel_parts) < 4:
            continue

        versjon      = rel_parts[0]          # 'v1' eller 'v2'
        klasse_mappe = rel_parts[1]          # 'Fast', 'Los', 'Idle'
        montering_id = rel_parts[2]          # 'Fast_1', 'Los_skrue_1og3' osv.

        # Ignorer Idle
        if "Idle" in klasse_mappe or "Idle" in montering_id:
            skipped_idle += 1
            continue

        df = pd.read_csv(csv_path)
        if not all(a in df.columns for a in AXES):
            continue

        signal_np = df[AXES].to_numpy()
        n = len(signal_np)

        if n < WINDOW_SAMPLES:
            skipped_short += 1
            continue

        # Bruker montering_id direkte som gruppe — ingen par-logikk
        gruppe_id = f"{versjon}_{montering_id}"

        for start in range(0, n - WINDOW_SAMPLES + 1, STEP_SAMPLES):
            window = signal_np[start : start + WINDOW_SAMPLES]
            w_df = pd.DataFrame(window, columns=AXES)
            w_df["id"]   = sample_id
            w_df["time"] = np.arange(WINDOW_SAMPLES)
            rows.append(w_df)

            labels[sample_id] = montering_id
            groups[sample_id] = gruppe_id
            sample_id += 1

        processed += 1

    print(f"  Prosessert: {processed} filer | Hoppet over: {skipped_idle} Idle, "
          f"{skipped_short} for korte")

    if not rows:
        raise ValueError("Ingen gyldige vinduer generert — sjekk mappestruktur og kolonnenavn.")

    long_df = pd.concat(rows, ignore_index=True)
    y = pd.Series(labels).sort_index()
    g = pd.Series(groups).sort_index()

    klasse_teller = y.apply(lambda l: "Fast" if "Fast" in l else "Los").value_counts()
    print(f"\nVindufordeling etter klasse:")
    for klasse, antall in klasse_teller.items():
        print(f"  {klasse}: {antall} vinduer")
    print(f"  Totalt: {len(y)} vinduer fra {y.nunique()} monteringer")
    print(f"  Antall unike grupper (for GroupShuffleSplit): {g.nunique()}\n")

    return long_df, y, g

# --------------------------------------------------------------------------- #
# 2. FEATURE-EKSTRAKSJON                                                      #
# --------------------------------------------------------------------------- #
def save_tsfresh_params(params, path: Path = Path("tsfresh_parametere.json")):
    output = {}
    for feature_name, param_list in sorted(params.items()):
        if param_list is None:
            output[feature_name] = {"antall_varianter": 1, "parametere": None}
        else:
            output[feature_name] = {
                "antall_varianter": len(param_list),
                "parametere": [dict(p) for p in param_list]
            }

    total_features_per_akse = sum(
        1 if v["parametere"] is None else v["antall_varianter"]
        for v in output.values()
    )
    wrapper = {
        "_info": {
            "antall_feature_typer": len(output),
            "antall_features_per_akse": total_features_per_akse,
            "antall_akser": len(AXES),
            "totalt_features": total_features_per_akse * len(AXES),
        },
        "features": output
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(wrapper, f, indent=2, ensure_ascii=False)
    print(f"✓ tsfresh-parametere lagret → {path}")
    print(f"  {len(output)} feature-typer → {total_features_per_akse} features/akse "
          f"× {len(AXES)} akser = {total_features_per_akse * len(AXES)} totalt")

def extract(long_df: pd.DataFrame) -> pd.DataFrame:
    print("\nEkstraherer features (tsfresh EfficientFCParameters)...")
    fc_params = EfficientFCParameters()
    save_tsfresh_params(fc_params)
    X = extract_features(
        long_df,
        column_id="id",
        column_sort="time",
        default_fc_parameters=fc_params,
        n_jobs=0,
        disable_progressbar=False
    )
    impute(X)
    return X

# --------------------------------------------------------------------------- #
# 3. MAJORITY VOTING                                                          #
# --------------------------------------------------------------------------- #
def majority_vote(y_true_montering, y_pred_vindu, montering_ids):
    df = pd.DataFrame({
        "montering": montering_ids,
        "sann":      y_true_montering,
        "pred":      y_pred_vindu
    })

    resultater = {}
    for montering, data in df.groupby("montering"):
        stemmer  = data["pred"].value_counts().to_dict()
        vinner   = data["pred"].value_counts().idxmax()
        sann     = data["sann"].iloc[0]
        resultater[montering] = {
            "sann":    sann,
            "pred":    vinner,
            "stemmer": stemmer,
            "riktig":  vinner == sann
        }
    return resultater

# --------------------------------------------------------------------------- #
# SAMLET RAPPORTFIGUR (CM + F1-tabell + ROC)                                 #
# --------------------------------------------------------------------------- #
def plot_full_report(
    y_true,
    y_pred,
    y_score,
    classes,
    title: str,
    subtitle: str,
    out_path: Path,
    cmap_cm: str = "Blues",
):
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=classes,
        zero_division=0,
    )

    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    has_roc = y_score is not None
    n_cols = 3 if has_roc else 2

    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    # ---- 1) CONFUSION MATRIX ----
    ax_cm = axes[0]
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap_cm,
        xticklabels=classes, yticklabels=classes,
        cbar=False, ax=ax_cm,
    )
    ax_cm.set_title("Confusion matrix")
    ax_cm.set_xlabel("Predikert")
    ax_cm.set_ylabel("Sann")

    # ---- 2) METRIKK-TABELL ----
    ax_tab = axes[1]
    ax_tab.axis("off")

    table_rows = []
    for i, cls in enumerate(classes):
        table_rows.append([
            cls, f"{prec[i]:.3f}", f"{rec[i]:.3f}", f"{f1[i]:.3f}", f"{int(support[i])}"
        ])

    table_rows.append(["accuracy", "", "", f"{accuracy:.3f}", f"{int(sum(support))}"])

    macro_p = float(np.mean(prec))
    macro_r = float(np.mean(rec))
    macro_f = float(np.mean(f1))
    table_rows.append(["macro avg", f"{macro_p:.3f}", f"{macro_r:.3f}", f"{macro_f:.3f}", f"{int(sum(support))}"])

    total = sum(support) if sum(support) > 0 else 1
    wp = float(np.sum(prec * support) / total)
    wr = float(np.sum(rec * support) / total)
    wf = float(np.sum(f1 * support) / total)
    table_rows.append(["weighted avg", f"{wp:.3f}", f"{wr:.3f}", f"{wf:.3f}", f"{int(sum(support))}"])

    tbl = ax_tab.table(
        cellText=table_rows,
        colLabels=["", "precision", "recall", "f1-score", "support"],
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.6)
    ax_tab.set_title("Classification report")

    # ---- 3) ROC-KURVE ----
    if has_roc:
        ax_roc = axes[2]
        y_true_bin = np.array([1 if v == classes[1] else 0 for v in y_true])
        fpr, tpr, _ = roc_curve(y_true_bin, y_score)
        roc_auc = auc(fpr, tpr)

        ax_roc.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
        ax_roc.plot([0, 1], [0, 1], linestyle="--", color="grey", lw=1)
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel("False positive rate")
        ax_roc.set_ylabel("True positive rate")
        ax_roc.set_title(f"ROC ({classes[1]} = positiv)")
        ax_roc.legend(loc="lower right")
        ax_roc.grid(alpha=0.3)

    fig.suptitle(f"{title} — {subtitle}", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"Samlet rapport lagret: {out_path}")

# --------------------------------------------------------------------------- #
# 4. RANDOM 80/20-EVALUERING                                                  #
# --------------------------------------------------------------------------- #
def train_and_evaluate_random_split(X, y, groups, title: str, n_runs: int = 20, test_size: float = 0.2):
    alle_sanne_vindu = []
    alle_pred_vindu  = []
    alle_score_vindu = []
    
    unique_classes   = ["Fast", "Los"]
    pos_class        = unique_classes[1]

    alle_mv_sanne = []
    alle_mv_pred  = []

    print(f"\nKjører GroupShuffleSplit 80/20 på monteringsnivå, {n_runs} ganger...\n")
    print(f"{'Runde':<7} {'Trening':>8} {'Test':>6} {'Vindu-acc':>10} {'Mont-acc':>9} {'#Mont':>6}")
    print("-" * 52)

    gss = GroupShuffleSplit(n_splits=n_runs, test_size=test_size, random_state=RANDOM_STATE)

    accuracies_vindu     = []
    accuracies_montering = []

    for runde, (train_idx, test_idx) in enumerate(gss.split(X, y, groups), start=1):
        X_train_raw = X.iloc[train_idx]
        X_test_raw  = X.iloc[test_idx]
        y_train_raw = y.iloc[train_idx]
        y_test_raw  = y.iloc[test_idx]

        y_train = y_train_raw.apply(lambda l: "Fast" if "Fast" in l else "Los")
        y_test  = y_test_raw.apply(lambda l: "Fast" if "Fast" in l else "Los")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            selector = SelectKBest(score_func=f_classif, k=min(TOP_K_FEATURES, X_train_raw.shape[1]))
            X_train_sel = selector.fit_transform(X_train_raw, y_train)
            X_test_sel  = selector.transform(X_test_raw)

        clf = RandomForestClassifier(
            n_estimators=100, n_jobs=-1,
            random_state=RANDOM_STATE, class_weight="balanced"
        )
        clf.fit(X_train_sel, y_train)
        y_pred = clf.predict(X_test_sel)

        proba = clf.predict_proba(X_test_sel)
        pos_idx = list(clf.classes_).index(pos_class)
        y_score = proba[:, pos_idx]

        acc_vindu = (y_pred == y_test.to_numpy()).mean()
        accuracies_vindu.append(acc_vindu)

        alle_sanne_vindu.extend(y_test.tolist())
        alle_pred_vindu.extend(y_pred.tolist())
        alle_score_vindu.extend(y_score.tolist())

        mv_resultater   = majority_vote(y_test, y_pred, y_test_raw)
        mv_sanne        = [v["sann"] for v in mv_resultater.values()]
        mv_pred         = [v["pred"] for v in mv_resultater.values()]
        acc_montering   = np.mean([v["riktig"] for v in mv_resultater.values()])
        accuracies_montering.append(acc_montering)

        alle_mv_sanne.extend(mv_sanne)
        alle_mv_pred.extend(mv_pred)

        n_test_mont = len(mv_resultater)
        print(f"  {runde:02d}    {len(X_train_raw):>8d} {len(X_test_raw):>6d} {acc_vindu:>10.4f} {acc_montering:>9.4f} {n_test_mont:>6d}")

    # ---- Vindu-nivå rapport ----
    print(f"\n{'='*60}")
    print(f"=== {title} — VINDU-NIVÅ (alle {n_runs} splittelser) ===")
    print(f"{'='*60}")
    print(classification_report(alle_sanne_vindu, alle_pred_vindu,
                                digits=3, labels=unique_classes))

    fname_vindu = PLOT_DIR / f"report_{title.replace(' ', '_')}_vindu.png"
    plot_full_report(
        y_true=alle_sanne_vindu,
        y_pred=alle_pred_vindu,
        y_score=alle_score_vindu,
        classes=unique_classes,
        title=title,
        subtitle=f"Vindu-nivå ({WINDOW_SECONDS}s, {n_runs} runder)",
        out_path=fname_vindu,
        cmap_cm="Blues"
    )

    # ---- Monteringsnivå rapport (majority voting) ----
    print(f"\n{'='*60}")
    print(f"=== {title} — MONTERINGSNIVÅ / MAJORITY VOTING (alle {n_runs} splittelser) ===")
    print(f"{'='*60}")
    print(classification_report(alle_mv_sanne, alle_mv_pred,
                                digits=3, labels=unique_classes))

    fname_mv = PLOT_DIR / f"report_{title.replace(' ', '_')}_montering.png"
    plot_full_report(
        y_true=alle_mv_sanne,
        y_pred=alle_mv_pred,
        y_score=None,
        classes=unique_classes,
        title=title,
        subtitle=f"Monteringsnivå, majority voting ({WINDOW_SECONDS}s, {n_runs} runder)",
        out_path=fname_mv,
        cmap_cm="Greens"
    )

    print(f"\nGjennomsnittlig vindu-accuracy:     {np.mean(accuracies_vindu):.4f} ± {np.std(accuracies_vindu):.4f}")
    print(f"Gjennomsnittlig montering-accuracy: {np.mean(accuracies_montering):.4f} ± {np.std(accuracies_montering):.4f}")

# --------------------------------------------------------------------------- #
# 5. ENDELIG MODELL — trent på ALL data                                      #
# --------------------------------------------------------------------------- #
def train_final_model(X, y, title: str):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nTrener endelig modell på alle {len(y)} vinduer...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selector = SelectKBest(score_func=f_classif, k=min(TOP_K_FEATURES, X.shape[1]))
        X_sel    = selector.fit_transform(X, y)

    clf = RandomForestClassifier(
        n_estimators=100, n_jobs=-1,
        random_state=RANDOM_STATE, class_weight="balanced"
    )
    clf.fit(X_sel, y)

    safe_title    = title.replace(" ", "_")
    selector_path = MODEL_DIR / f"{safe_title}_selector.joblib"
    model_path    = MODEL_DIR / f"{safe_title}_model.joblib"

    joblib.dump(selector, selector_path)
    joblib.dump(clf,      model_path)

    print(f"✓ Selector lagret → {selector_path}")
    print(f"✓ Modell lagret   → {model_path}")

    feature_names   = X.columns[selector.get_support()]
    feature_scores  = selector.scores_[selector.get_support()]
    topp_df = (
        pd.DataFrame({"feature": feature_names, "F_score": feature_scores})
        .sort_values("F_score", ascending=False)
        .reset_index(drop=True)
    )
    topp_df.index += 1

    print(f"\nTopp 10 av {TOP_K_FEATURES} features (F-score, SelectKBest):")
    print(topp_df.head(10).to_string())

    impuls_moenstre = ["kurtosis", "c3", "cid_ce", "abs_energy",
                       "absolute_sum_of_changes", "skewness"]
    print(f"\nImpulsivitets-features i topp {TOP_K_FEATURES} (vindu = {WINDOW_SECONDS}s):")
    for moenster in impuls_moenstre:
        treff = topp_df[topp_df["feature"].str.contains(moenster, case=False, regex=False)]
        if len(treff) > 0:
            beste = treff.iloc[0]
            print(f"  {moenster:25s} | antall: {len(treff):3d} | beste rang: {treff.index.min():3d} "
                  f"| F={beste['F_score']:.1f}")
        else:
            print(f"  {moenster:25s} | ikke blant topp {TOP_K_FEATURES}")

    topp_path = MODEL_DIR / f"{safe_title}_topp_features.csv"
    topp_df.to_csv(topp_path, index=True)
    print(f"\n✓ Alle {TOP_K_FEATURES} features lagret → {topp_path}")
    return selector, clf

# --------------------------------------------------------------------------- #
# MAIN                                                                        #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    cached = load_cache(DATA_ROOT)
    if cached:
        long_df, y, groups = cached[:3]
        X = cached[-1]
    else:
        long_df, y, groups = load_and_window(DATA_ROOT)
        X = extract(long_df)
        save_cache(DATA_ROOT, long_df, y, groups, X)

    def kategoriser(label):
        if "Fast" in label: return "Fast"
        if "Los"  in label: return "Los"
        return "Idle"

    y_kat  = y.apply(kategoriser)
    mask   = y_kat != "Idle"

    X_bin      = X.loc[mask.values]
    y_bin      = y[mask.values]
    groups_bin = groups[mask.values]

    train_and_evaluate_random_split(X_bin, y_bin, groups_bin, "Binaer_Fast_vs_Los", n_runs=20, test_size=0.2)

    y_bin_klasse = y_bin.apply(lambda l: "Fast" if "Fast" in l else "Los")
    train_final_model(X_bin, y_bin_klasse, "Binaer_Fast_vs_Los")

    print("\nAnalyse ferdig. Sjekk plots-mappen for samlerapporter.")