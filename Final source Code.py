"""
CPRI Hackathon — End-to-End Source code
=====================================
Connects the team's separate notebooks/scripts into one automated run:

    Load training data
        -> Clean data                       (Preprocessing_data.ipynb)
        -> Feature engineering               (Feature_Engineering.ipynb, now actually
                                               wired into training instead of just printed)
        -> Train models                      (Untitled2.ipynb regressor +
                                               validity_pipeline.py classifier)
    Load test data
        -> Predict Reference_Parameter
        -> Predict VALID / INVALID
        -> Generate <TeamName>.csv
        -> Generate summary.json

Nothing here hand-edits individual rows: duplicate removal, outlier flags,
imputation, engineered-feature selection, and model selection are all rule-
or CV-based, so the whole thing can be re-run on a fresh copy of the raw
Excel file with no manual intervention.

Usage:
    python final_pipeline.py \
        --input CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx \
        --team-name YourTeamName \
        --output-dir outputs
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    root_mean_squared_error,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Shared column definitions (from Preprocessing_data.ipynb)
# ---------------------------------------------------------------------------
INPUT_COLS = [
    "Applied_Voltage_kV",
    "Load_Current_A",
    "Ambient_Temperature_C",
    "Test_Duration_min",
]
SENSOR_COLS = ["Sensor_S1", "Sensor_S2", "Sensor_S3", "Sensor_S4"]
FEATURE_COLS = INPUT_COLS + SENSOR_COLS  # used for duplicate detection

FLAG_COLS = (
    [f"{c}_negative_flag" for c in SENSOR_COLS]
    + [f"{c}_outlier_flag" for c in SENSOR_COLS]
    + ["Sensor_Fault_Flag"]
    + [f"{c}_was_missing" for c in SENSOR_COLS]
    + ["Total_Missing_Sensors"]
)
BASELINE_FEATURES = INPUT_COLS + SENSOR_COLS + FLAG_COLS + ["Power_kW"]

# Candidate engineered features (from Feature_Engineering.ipynb)
CANDIDATE_FEATURES = [
    "VxI", "I2", "V2", "IxDur", "S1_S2", "S1_S3", "S2_S3",
    "Sensor_avg", "Sensor_max", "Sensor_min",
]
KEEP_THRESHOLD_F1 = 0.01
KEEP_THRESHOLD_R2 = 0.01
FE_SEEDS = [0, 1, 2, 3, 4]
FE_SPLITS = 5


def log(msg=""):
    print(msg, flush=True)


def section(title):
    log("\n" + "=" * 70)
    log(title)
    log("=" * 70)


# ===========================================================================
# STAGE 1 — LOAD
# ===========================================================================
def load_raw(path):
    train = pd.read_excel(path, sheet_name="Training_Data")
    test = pd.read_excel(path, sheet_name="Test_Data")
    log(f"Loaded Training_Data: {train.shape}")
    log(f"Loaded Test_Data: {test.shape}")
    return train, test


# ===========================================================================
# STAGE 2 — CLEAN  (Preprocessing_data.ipynb)
# ===========================================================================
def drop_duplicate_pairs(df):
    before = len(df)
    dup_mask = df.duplicated(subset=FEATURE_COLS, keep="first")
    df = df.loc[~dup_mask].reset_index(drop=True)
    log(f"Duplicate measurement pairs found: {dup_mask.sum()} "
        f"(dropped {before - len(df)}, remaining {len(df)})")
    return df


def flag_sensor_faults(df, bounds=None):
    df = df.copy()
    computed_bounds = {}
    fault_flag = pd.Series(0, index=df.index)

    for col in SENSOR_COLS:
        neg_mask = df[col] < 0
        df[f"{col}_negative_flag"] = neg_mask.astype(int)

        if bounds is None:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        else:
            lo, hi = bounds[col]
        computed_bounds[col] = (lo, hi)

        out_mask = (df[col] < lo) | (df[col] > hi)
        df[f"{col}_outlier_flag"] = out_mask.astype(int)

        fault_flag = fault_flag | neg_mask.astype(int) | out_mask.astype(int)

    df["Sensor_Fault_Flag"] = fault_flag
    log(f"Rows flagged with at least one sensor fault: {fault_flag.sum()}")
    return df, computed_bounds


def impute_missing(df, medians=None):
    df = df.copy()
    if medians is None:
        medians = df[SENSOR_COLS].median()

    for col in SENSOR_COLS:
        df[f"{col}_was_missing"] = df[col].isna().astype(int)

    df[SENSOR_COLS] = df[SENSOR_COLS].fillna(medians)
    return df, medians


def encode_label(df):
    df = df.copy()
    df["Validity_Label_Bin"] = df["Validity_Label"].map({"Valid": 1, "Invalid": 0})
    return df


def add_base_features(df):
    df = df.copy()
    df["Power_kW"] = df["Applied_Voltage_kV"] * df["Load_Current_A"]
    missing_cols = [c for c in df.columns if c.endswith("_was_missing")]
    df["Total_Missing_Sensors"] = df[missing_cols].sum(axis=1)
    return df


def clean_pipeline(train, test):
    train = drop_duplicate_pairs(train)
    train, bounds = flag_sensor_faults(train, bounds=None)
    train, medians = impute_missing(train, medians=None)
    train = encode_label(train)
    train = add_base_features(train)

    # test uses TRAIN bounds/medians -> no leakage
    test, _ = flag_sensor_faults(test, bounds=bounds)
    test, _ = impute_missing(test, medians=medians)
    test = add_base_features(test)

    log(f"Cleaned training shape: {train.shape} | cleaned test shape: {test.shape}")
    return train, test


# ===========================================================================
# STAGE 3 — FEATURE ENGINEERING  (Feature_Engineering.ipynb, now wired in)
# ===========================================================================
def add_candidate_features(df):
    d = df.copy()
    d["VxI"] = d["Applied_Voltage_kV"] * d["Load_Current_A"]  # == Power_kW
    d["I2"] = d["Load_Current_A"] ** 2
    d["V2"] = d["Applied_Voltage_kV"] ** 2
    d["IxDur"] = d["Load_Current_A"] * d["Test_Duration_min"]
    d["S1_S2"] = d["Sensor_S1"] - d["Sensor_S2"]
    d["S1_S3"] = d["Sensor_S1"] - d["Sensor_S3"]
    d["S2_S3"] = d["Sensor_S2"] - d["Sensor_S3"]
    sensors = d[SENSOR_COLS]
    d["Sensor_avg"] = sensors.mean(axis=1)
    d["Sensor_max"] = sensors.max(axis=1)
    d["Sensor_min"] = sensors.min(axis=1)
    return d


def _cv_score_classifier(df, feats, y):
    X = df[feats].values
    scores = []
    for seed in FE_SEEDS:
        skf = StratifiedKFold(n_splits=FE_SPLITS, shuffle=True, random_state=seed)
        for tr, va in skf.split(X, y):
            clf = GradientBoostingClassifier(random_state=RANDOM_STATE)
            clf.fit(X[tr], y[tr])
            pred = clf.predict(X[va])
            scores.append(f1_score(y[va], pred, pos_label=1, zero_division=0))
    return np.array(scores)


def _cv_score_regressor(df, feats, y):
    X = df[feats].values
    r2s, maes = [], []
    for seed in FE_SEEDS:
        kf = KFold(n_splits=FE_SPLITS, shuffle=True, random_state=seed)
        for tr, va in kf.split(X):
            reg = GradientBoostingRegressor(random_state=RANDOM_STATE)
            reg.fit(X[tr], y[tr])
            pred = reg.predict(X[va])
            maes.append(mean_absolute_error(y[va], pred))
            r2s.append(1 - np.sum((y[va] - pred) ** 2) / np.sum((y[va] - y[va].mean()) ** 2))
    return np.array(r2s), np.array(maes)


def select_engineered_features(train_fe):
    """Paired CV comparison: baseline vs baseline+1 candidate. Only keep
    candidates that clearly beat baseline (mean delta above threshold)."""
    section("STAGE 3: FEATURE ENGINEERING — automatic candidate selection")

    y_clf = (train_fe["Validity_Label"] == "Invalid").astype(int).values
    y_reg = train_fe["Reference_Parameter"].values

    base_f1 = _cv_score_classifier(train_fe, BASELINE_FEATURES, y_clf)
    base_r2, base_mae = _cv_score_regressor(train_fe, BASELINE_FEATURES, y_reg)
    log(f"Baseline classifier F1(Invalid): {base_f1.mean():.4f}")
    log(f"Baseline regressor  R2: {base_r2.mean():.4f} | MAE: {base_mae.mean():.4f}")

    rows = []
    for feat in CANDIDATE_FEATURES:
        feats = BASELINE_FEATURES + [feat]
        f1 = _cv_score_classifier(train_fe, feats, y_clf)
        r2, mae = _cv_score_regressor(train_fe, feats, y_reg)
        rows.append({
            "feature": feat,
            "delta_F1": f1.mean() - base_f1.mean(),
            "delta_R2": r2.mean() - base_r2.mean(),
            "delta_MAE": mae.mean() - base_mae.mean(),
        })
    res_df = pd.DataFrame(rows).sort_values("delta_R2", ascending=False)
    log("\nCandidate feature deltas vs baseline:")
    log(res_df.to_string(index=False))

    keep_for_reg = res_df.loc[res_df["delta_R2"] > KEEP_THRESHOLD_R2, "feature"].tolist()
    keep_for_clf = res_df.loc[res_df["delta_F1"] > KEEP_THRESHOLD_F1, "feature"].tolist()

    log(f"\nFeatures kept for REGRESSOR (dR2 > {KEEP_THRESHOLD_R2}): {keep_for_reg or '(none)'}")
    log(f"Features kept for CLASSIFIER (dF1 > {KEEP_THRESHOLD_F1}): {keep_for_clf or '(none)'}")

    diagnostics = {
        "baseline_classifier_f1_invalid": round(float(base_f1.mean()), 4),
        "baseline_regressor_r2": round(float(base_r2.mean()), 4),
        "baseline_regressor_mae": round(float(base_mae.mean()), 4),
        "candidate_deltas": res_df.round(4).to_dict(orient="records"),
        "kept_for_regressor": keep_for_reg,
        "kept_for_classifier": keep_for_clf,
    }
    return keep_for_reg, keep_for_clf, diagnostics


# ===========================================================================
# STAGE 4a — TRAIN REGRESSOR  (Untitled2.ipynb, target = Reference_Parameter)
# ===========================================================================
def train_regressor(train_fe, feature_cols):
    section("STAGE 4a: TRAIN MODELS — Reference_Parameter regressor")

    X = train_fe[feature_cols]
    y = train_fe["Reference_Parameter"]

    models = {
        "Linear Regression": Pipeline([("imputer", SimpleImputer(strategy="median")),
                                        ("model", LinearRegression())]),
        "Random Forest": Pipeline([("imputer", SimpleImputer(strategy="median")),
                                    ("model", RandomForestRegressor(random_state=RANDOM_STATE))]),
        "Extra Trees": Pipeline([("imputer", SimpleImputer(strategy="median")),
                                  ("model", ExtraTreesRegressor(random_state=RANDOM_STATE))]),
        "Gradient Boosting": Pipeline([("imputer", SimpleImputer(strategy="median")),
                                        ("model", GradientBoostingRegressor(random_state=RANDOM_STATE))]),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    comparison = {}
    best_name, best_mae, best_rmse = None, float("inf"), None

    for name, model in models.items():
        cv = cross_validate(
            model, X, y, cv=kf,
            scoring={"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"},
        )
        mae = -cv["test_mae"].mean()
        rmse = -cv["test_rmse"].mean()
        comparison[name] = {"MAE": round(mae, 4), "RMSE": round(rmse, 4)}
        log(f"  {name:<22} MAE={mae:.4f}  RMSE={rmse:.4f}")
        if mae < best_mae:
            best_name, best_mae, best_rmse = name, mae, rmse

    best_model = models[best_name]
    best_model.fit(X, y)

    perm = permutation_importance(best_model, X, y, n_repeats=5, random_state=RANDOM_STATE)
    order = perm.importances_mean.argsort()[::-1]
    top_features = [
        {"feature": feature_cols[i], "importance": round(float(perm.importances_mean[i]), 4)}
        for i in order[:10]
    ]

    log(f"\nBest regressor: {best_name} (MAE={best_mae:.4f}, RMSE={best_rmse:.4f})")

    metrics = {
        "best_model": best_name,
        "validation_mae": round(best_mae, 4),
        "validation_rmse": round(best_rmse, 4),
        "model_comparison": comparison,
        "top_features": top_features,
    }
    return best_model, metrics


# ===========================================================================
# STAGE 4b — TRAIN CLASSIFIER  (validity_pipeline.py, target = Validity_Label)
# ===========================================================================
def train_classifier(train_fe, feature_cols):
    section("STAGE 4b: TRAIN MODELS — Validity (VALID/INVALID) classifier")

    X = train_fe[feature_cols].copy()
    y = train_fe["Validity_Label_Bin"]  # 1 = Valid, 0 = Invalid

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    sample_weight_train = compute_sample_weight("balanced", y_train)

    candidates = {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
            X_train_scaled, X_val_scaled, False,
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE),
            X_train, X_val, False,
        ),
        "Extra Trees": (
            ExtraTreesClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE),
            X_train, X_val, False,
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE),
            X_train, X_val, True,
        ),
    }

    comparison, fitted, val_preds = {}, {}, {}
    for name, (model, xtr, xval, use_sw) in candidates.items():
        if use_sw:
            model.fit(xtr, y_train, sample_weight=sample_weight_train)
        else:
            model.fit(xtr, y_train)
        preds = model.predict(xval)
        fitted[name] = model
        val_preds[name] = preds

        precision = precision_score(y_val, preds, pos_label=0, zero_division=0)
        recall = recall_score(y_val, preds, pos_label=0, zero_division=0)
        f1_invalid = f1_score(y_val, preds, pos_label=0, zero_division=0)
        f1_macro = f1_score(y_val, preds, average="macro")

        comparison[name] = {
            "precision_invalid": round(precision, 4),
            "recall_invalid": round(recall, 4),
            "f1_invalid": round(f1_invalid, 4),
            "f1_macro": round(f1_macro, 4),
        }
        log(f"  {name:<22} P={precision:.4f} R={recall:.4f} "
            f"F1(Invalid)={f1_invalid:.4f} F1(macro)={f1_macro:.4f}")

    best_name = max(comparison, key=lambda n: comparison[n]["f1_invalid"])
    best_f1_invalid = comparison[best_name]["f1_invalid"]
    log(f"\nBest classifier: {best_name} (F1(Invalid)={best_f1_invalid:.4f})")
    log("\nClassification report (validation split, best model):")
    log(classification_report(y_val, val_preds[best_name], target_names=["Invalid", "Valid"]))

    # Refit best model on ALL training data for final predictions
    if best_name == "Logistic Regression":
        full_scaler = StandardScaler().fit(X)
        X_full_scaled = full_scaler.transform(X)
        final_model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
        final_model.fit(X_full_scaled, y)
        predict_fn = lambda df_feats: final_model.predict(full_scaler.transform(df_feats))
    elif best_name == "Gradient Boosting":
        sw_full = compute_sample_weight("balanced", y)
        final_model = GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE)
        final_model.fit(X, y, sample_weight=sw_full)
        predict_fn = lambda df_feats: final_model.predict(df_feats)
    else:
        ModelClass = RandomForestClassifier if best_name == "Random Forest" else ExtraTreesClassifier
        final_model = ModelClass(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE)
        final_model.fit(X, y)
        predict_fn = lambda df_feats: final_model.predict(df_feats)

    if hasattr(final_model, "feature_importances_"):
        importances = pd.Series(final_model.feature_importances_, index=feature_cols)
    elif hasattr(final_model, "coef_"):
        importances = pd.Series(np.abs(final_model.coef_[0]), index=feature_cols)
    else:
        importances = pd.Series(dtype=float)
    top_features = [
        {"feature": f, "importance": round(float(v), 4)}
        for f, v in importances.sort_values(ascending=False).head(10).items()
    ]

    metrics = {
        "best_model": best_name,
        "f1_invalid": best_f1_invalid,
        "model_comparison": comparison,
        "top_features": top_features,
    }
    return predict_fn, metrics


# ===========================================================================
# STAGE 9 — TASK 03: AUTOMATED TEST SUMMARY
# ===========================================================================
# Fixed, pre-written explanation of the team's approach. Kept under the
# 100-word limit required by the task; word count is asserted at runtime
# so the script fails loudly if this text is ever edited past the limit.
APPROACH_EXPLANATION = (
    "Our pipeline first cleans the raw sensor data by removing duplicate "
    "measurements, flagging physically impossible or statistically outlying "
    "sensor readings, and imputing missing values using training-set medians "
    "to avoid leakage. We then engineer candidate features (power, squared "
    "terms, sensor differences and aggregates) and keep only those that "
    "measurably improve cross-validated performance. Two models are trained: "
    "a regressor predicting Reference_Parameter and a classifier predicting "
    "Valid/Invalid records, each chosen from several candidate algorithms by "
    "cross-validation. The best models are applied to the test set to "
    "generate predictions, flag abnormal records, and produce this automated "
    "summary."
)
_word_count = len(APPROACH_EXPLANATION.split())
assert _word_count <= 100, f"Approach explanation is {_word_count} words (>100)."


def generate_test_summary(test_fe, train_fe, team_name, output_dir):
    """Task 03 — build the required automated test summary:
      - number of records analysed
      - number of abnormal/invalid records identified
      - min & max predicted Reference_Parameter
      - average predicted Reference_Parameter
      - 3 Test_IDs requiring the highest attention
      - <=100-word explanation of the team's approach
    Writes both a human-readable .txt and a machine-readable .json copy.
    """
    section("STAGE 9: TASK 03 — AUTOMATED TEST SUMMARY")

    n_records = int(len(test_fe))
    invalid_mask = test_fe["Predicted_Validity"] == "Invalid"
    n_invalid = int(invalid_mask.sum())

    ref_pred = test_fe["Predicted_Reference_Parameter"]
    ref_min = float(ref_pred.min())
    ref_max = float(ref_pred.max())
    ref_avg = float(ref_pred.mean())

    # "Highest attention" = predicted-Invalid records ranked by how far their
    # predicted Reference_Parameter sits from the TRAINING distribution's mean
    # (in training std-devs). Invalid records are prioritised over Valid ones;
    # ties/insufficient-Invalid cases are backfilled by the same extremity
    # score so there are always 3 IDs even if <3 records were flagged Invalid.
    train_mean = float(train_fe["Reference_Parameter"].mean())
    train_std = float(train_fe["Reference_Parameter"].std()) or 1.0
    extremity = (ref_pred - train_mean).abs() / train_std
    attention_score = np.where(invalid_mask, 1000.0, 0.0) + extremity
    top3_order = pd.Series(attention_score, index=test_fe.index).sort_values(ascending=False)
    top3_idx = top3_order.index[:3]
    top3_ids = test_fe.loc[top3_idx, "Test_ID"].astype(str).tolist()

    summary = {
        "team_name": team_name,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "records_analysed": n_records,
        "abnormal_invalid_records": n_invalid,
        "reference_parameter_predicted": {
            "min": round(ref_min, 4),
            "max": round(ref_max, 4),
            "average": round(ref_avg, 4),
        },
        "test_ids_requiring_highest_attention": top3_ids,
        "approach_explanation": APPROACH_EXPLANATION,
        "approach_explanation_word_count": _word_count,
    }

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "Test_Summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    txt_path = os.path.join(output_dir, "Test_Summary.txt")
    with open(txt_path, "w") as f:
        f.write("AUTOMATED TEST SUMMARY\n")
        f.write("=" * 40 + "\n")
        f.write(f"Team: {team_name}\n")
        f.write(f"Generated: {summary['generated_at']}\n\n")
        f.write(f"Number of records analysed: {n_records}\n")
        f.write(f"Number of abnormal/invalid records identified: {n_invalid}\n\n")
        f.write("Predicted Reference_Parameter:\n")
        f.write(f"  Minimum: {ref_min:.4f}\n")
        f.write(f"  Maximum: {ref_max:.4f}\n")
        f.write(f"  Average: {ref_avg:.4f}\n\n")
        f.write("Test IDs requiring the highest attention:\n")
        for tid in top3_ids:
            f.write(f"  - {tid}\n")
        f.write(f"\nApproach ({_word_count} words):\n{APPROACH_EXPLANATION}\n")

    log(f"Records analysed: {n_records}")
    log(f"Abnormal/Invalid records: {n_invalid}")
    log(f"Reference_Parameter — min={ref_min:.4f}, max={ref_max:.4f}, avg={ref_avg:.4f}")
    log(f"Top-3 attention Test_IDs: {top3_ids}")
    log(f"Saved: {json_path}")
    log(f"Saved: {txt_path}")

    return summary, json_path, txt_path


# ===========================================================================
# STAGE 5/6 — PREDICT + STAGE 7/8 — OUTPUTS
# ===========================================================================
def build_outputs(test_fe, reg_model, reg_feature_cols, clf_predict_fn, clf_feature_cols,
                   reg_metrics, clf_metrics, fe_diagnostics, team_name, output_dir,
                   train_shape, test_shape, elapsed_seconds, train_fe):
    section("STAGE 5/6: PREDICT on test data")

    test_fe = test_fe.copy()
    test_fe["Predicted_Reference_Parameter"] = reg_model.predict(test_fe[reg_feature_cols])
    clf_preds = clf_predict_fn(test_fe[clf_feature_cols])
    test_fe["Predicted_Validity"] = np.where(clf_preds == 1, "Valid", "Invalid")

    pred_counts = test_fe["Predicted_Validity"].value_counts().to_dict()
    log(f"Predicted Reference_Parameter — mean={test_fe['Predicted_Reference_Parameter'].mean():.4f}, "
        f"std={test_fe['Predicted_Reference_Parameter'].std():.4f}")
    log(f"Predicted validity distribution: {pred_counts}")

    section("STAGE 7: GENERATE SUBMISSION CSV")
    os.makedirs(output_dir, exist_ok=True)
    submission = test_fe[["Test_ID", "Predicted_Reference_Parameter", "Predicted_Validity"]].copy()
    submission_path = os.path.join(output_dir, f"{team_name}.csv")
    submission.to_csv(submission_path, index=False)
    log(f"Saved: {submission_path}  (shape={submission.shape})")

    section("STAGE 8: GENERATE PIPELINE SUMMARY JSON")
    task03_summary, task03_json_path, task03_txt_path = generate_test_summary(
        test_fe, train_fe, team_name, output_dir
    )

    summary = {
        "team_name": team_name,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_seconds": round(elapsed_seconds, 1),
        "data": {
            "raw_training_shape": list(train_shape),
            "raw_test_shape": list(test_shape),
            "cleaned_test_rows_predicted": int(len(test_fe)),
        },
        "feature_engineering": fe_diagnostics,
        "reference_parameter_model": reg_metrics,
        "validity_model": clf_metrics,
        "predictions": {
            "validity_distribution": {str(k): int(v) for k, v in pred_counts.items()},
            "reference_parameter_mean": round(float(test_fe["Predicted_Reference_Parameter"].mean()), 4),
            "reference_parameter_std": round(float(test_fe["Predicted_Reference_Parameter"].std()), 4),
        },
        "task03_test_summary": task03_summary,
        "outputs": {
            "submission_csv": submission_path,
            "task03_summary_json": task03_json_path,
            "task03_summary_txt": task03_txt_path,
        },
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"Saved: {summary_path}")

    return submission_path, summary_path, task03_json_path, task03_txt_path


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="End-to-end CPRI hackathon pipeline")
    parser.add_argument("--input", default="CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx",
                         help="Path to the raw Excel file (must contain Training_Data and Test_Data sheets)")
    parser.add_argument("--team-name", default="Astra Watt",
                         help="Used to name the submission file: <TeamName>.csv")
    parser.add_argument("--output-dir", default="outputs",
                         help="Directory where TeamName.csv and summary.json are written")
    args = parser.parse_args()

    start = time.time()

    section("STAGE 1: LOAD TRAINING + TEST DATA")
    train_raw, test_raw = load_raw(args.input)

    section("STAGE 2: CLEAN DATA")
    train_clean, test_clean = clean_pipeline(train_raw, test_raw)

    train_fe = add_candidate_features(train_clean)
    test_fe = add_candidate_features(test_clean)

    keep_for_reg, keep_for_clf, fe_diag = select_engineered_features(train_fe)
    reg_feature_cols = BASELINE_FEATURES + keep_for_reg
    clf_feature_cols = BASELINE_FEATURES + keep_for_clf

    reg_model, reg_metrics = train_regressor(train_fe, reg_feature_cols)
    clf_predict_fn, clf_metrics = train_classifier(train_fe, clf_feature_cols)

    elapsed = time.time() - start
    build_outputs(
        test_fe, reg_model, reg_feature_cols, clf_predict_fn, clf_feature_cols,
        reg_metrics, clf_metrics, fe_diag, args.team_name, args.output_dir,
        train_raw.shape, test_raw.shape, elapsed, train_fe,
    )

    section("DONE")
    log(f"Total runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
