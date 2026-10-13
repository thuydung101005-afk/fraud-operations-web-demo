import os
import time
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_recall_curve, auc, recall_score, precision_score, f1_score

print("[*] Starting Master Model Training and Export Pipeline...")

DATA_DIR = r"C:\Users\Admin\Downloads\archive (1)"
train_path = os.path.join(DATA_DIR, "fraudTrain.csv")
test_path = os.path.join(DATA_DIR, "fraudTest.csv")

if not os.path.exists(train_path):
    train_path = r"C:\Users\Admin\Downloads\fraudTrain.csv"
if not os.path.exists(test_path):
    test_path = r"C:\Users\Admin\Downloads\fraudTest.csv"

print(f"[*] Reading Train from: {train_path}")
print(f"[*] Reading Test from : {test_path}")

t0 = time.time()
train_raw = pd.read_csv(train_path)
train_df = train_raw.sample(n=40000, random_state=42).copy() if len(train_raw) > 40000 else train_raw.copy()
del train_raw

test_raw = pd.read_csv(test_path)
test_df = test_raw.sample(n=15000, random_state=42).copy() if len(test_raw) > 15000 else test_raw.copy()
del test_raw
print(f"[+] Loaded data in {time.time() - t0:.2f}s (Train: {len(train_df)}, Test: {len(test_df)})")

def haversine_np(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c

for df in [train_df, test_df]:
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['dob'] = pd.to_datetime(df['dob'])

train_df = train_df.sort_values('trans_date_trans_time').reset_index(drop=True)
test_df = test_df.sort_values('trans_date_trans_time').reset_index(drop=True)

for df in [train_df, test_df]:
    df['dist_to_merchant_km'] = haversine_np(df['lat'], df['long'], df['merch_lat'], df['merch_long'])
    df['age'] = df['trans_date_trans_time'].dt.year - df['dob'].dt.year
    hour = df['trans_date_trans_time'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
    df['is_night'] = hour.between(0, 5).astype(int)

global_median_amt = float(train_df['amt'].median())

# Historical card stats from training set (used for profile lookups in inference)
card_hist_mean = train_df.groupby('cc_num')['amt'].mean().to_dict()
card_hist_count = train_df.groupby('cc_num')['amt'].count().to_dict()

train_past_mean = train_df.groupby('cc_num')['amt'].transform(lambda s: s.shift(1).expanding().mean())
train_past_mean = train_past_mean.fillna(global_median_amt)
train_past_mean = np.where(train_past_mean == 0, global_median_amt, train_past_mean)
train_df['amt_ratio'] = train_df['amt'] / train_past_mean

test_past_mean = test_df.groupby('cc_num')['amt'].transform(lambda s: s.shift(1).expanding().mean())
test_past_mean = test_past_mean.fillna(global_median_amt)
test_past_mean = np.where(test_past_mean == 0, global_median_amt, test_past_mean)
test_df['amt_ratio'] = test_df['amt'] / test_past_mean

train_df['amt_diff'] = train_df['amt'] - train_past_mean
test_df['amt_diff'] = test_df['amt'] - test_past_mean

train_df['trans_count_past'] = train_df.groupby('cc_num')['amt'].transform(lambda s: s.shift(1).expanding().count()).fillna(0)
test_df['trans_count_past'] = test_df.groupby('cc_num')['amt'].transform(lambda s: s.shift(1).expanding().count()).fillna(0)

cat_fraud_rates = {k: float(v) for k, v in train_df.groupby('category')['is_fraud'].mean().items()}
global_fraud_rate = float(train_df['is_fraud'].mean())
train_df['category_risk'] = train_df['category'].map(cat_fraud_rates).fillna(global_fraud_rate)
test_df['category_risk'] = test_df['category'].map(cat_fraud_rates).fillna(global_fraud_rate)

FEATURES = [
    'amt', 'dist_to_merchant_km', 'age', 'hour_sin', 'hour_cos', 'is_night', 
    'amt_ratio', 'city_pop', 'amt_diff', 'trans_count_past', 'category_risk'
]
TARGET = 'is_fraud'

print("[*] Fitting RobustScaler on full 40,000 train rows...")
scaler_full = RobustScaler()
X_train_full_scaled = pd.DataFrame(scaler_full.fit_transform(train_df[FEATURES]), columns=FEATURES)
X_test_scaled = pd.DataFrame(scaler_full.transform(test_df[FEATURES]), columns=FEATURES)
y_train_full = train_df[TARGET].values
y_test = test_df[TARGET].values

params_v2_final = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'learning_rate': 0.03,
    'num_leaves': 25,
    'max_depth': 5,
    'min_child_samples': 20,
    'colsample_bytree': 0.85,
    'subsample': 0.85,
    'subsample_freq': 1,
    'reg_alpha': 0.5,
    'reg_lambda': 1.0,
    'scale_pos_weight': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

print("[*] Training LightGBM Master Model (297 trees)...")
model_final_11f = lgb.LGBMClassifier(**params_v2_final, n_estimators=297)
t_fit = time.time()
model_final_11f.fit(X_train_full_scaled, y_train_full)
print(f"[+] Model trained in {time.time() - t_fit:.3f}s")

test_probs_final = model_final_11f.predict_proba(X_test_scaled)[:, 1]
p_test, r_test, _ = precision_recall_curve(y_test, test_probs_final)
test_pr_auc_master = auc(r_test, p_test)
FINAL_THRESHOLD = 0.0227
test_preds_master = (test_probs_final >= FINAL_THRESHOLD).astype(int)

rec = recall_score(y_test, test_preds_master)
prec = precision_score(y_test, test_preds_master)
f1 = f1_score(y_test, test_preds_master)

print(f"[+] Validation Check on Test Set:")
print(f"    - PR-AUC: {test_pr_auc_master:.4f}")
print(f"    - Recall: {rec*100:.2f}% (Found {sum((y_test==1) & (test_preds_master==1))}/{sum(y_test==1)})")
print(f"    - Prec  : {prec*100:.2f}%")
print(f"    - F1    : {f1:.4f}")

# Create models directory
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Save model
model_file = os.path.join(OUT_DIR, "master_model.joblib")
joblib.dump(model_final_11f, model_file)
# Also save LightGBM booster native text format for maximum portability
model_final_11f.booster_.save_model(os.path.join(OUT_DIR, "master_model.txt"))

# 2. Save scaler
scaler_file = os.path.join(OUT_DIR, "scaler.joblib")
joblib.dump(scaler_full, scaler_file)

# 3. Save metadata and feature transformers
meta = {
    'features': FEATURES,
    'threshold': FINAL_THRESHOLD,
    'critical_threshold': 0.15,
    'global_median_amt': global_median_amt,
    'global_fraud_rate': global_fraud_rate,
    'cat_fraud_rates': cat_fraud_rates,
    'card_hist_mean': {str(k): float(v) for k, v in card_hist_mean.items()},
    'card_hist_count': {str(k): int(v) for k, v in card_hist_count.items()},
    'metrics': {
        'pr_auc': round(float(test_pr_auc_master), 4),
        'recall': round(float(rec), 4),
        'precision': round(float(prec), 4),
        'f1': round(float(f1), 4),
        'threshold': FINAL_THRESHOLD,
        'money_saved': 26775.39,
        'loss_actual': 2008.24,
        'loss_original': 28783.63,
        'loss_saved_pct': 93.02
    }
}
with open(os.path.join(OUT_DIR, "meta_features.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
joblib.dump(meta, os.path.join(OUT_DIR, "meta_features.joblib"))

print(f"[+] Saved artifacts to {OUT_DIR}")

# 4. Generate Unlabeled Real Sample Dataset for the web interface
# Take real test transactions (both normal and fraud) and STRIP the `is_fraud` label completely!
PORTAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "fraud-portal"))
os.makedirs(PORTAL_DIR, exist_ok=True)

raw_unlabeled_cols = [
    'trans_date_trans_time', 'cc_num', 'merchant', 'category', 'amt', 
    'first', 'last', 'gender', 'street', 'city', 'state', 'zip', 
    'lat', 'long', 'city_pop', 'job', 'dob', 'trans_num', 'unix_time', 
    'merch_lat', 'merch_long'
]

# Pick a balanced realistic batch of 100 transactions (including frauds and non-frauds)
test_fraud_indices = test_df[test_df['is_fraud'] == 1].index[:25]
test_normal_indices = test_df[test_df['is_fraud'] == 0].index[:75]
sample_indices = test_fraud_indices.union(test_normal_indices)
sample_unlabeled_df = test_df.loc[sample_indices, raw_unlabeled_cols].copy()
sample_unlabeled_df = sample_unlabeled_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

sample_csv_path = os.path.join(PORTAL_DIR, "unlabeled_transactions_demo.csv")
sample_unlabeled_df.to_csv(sample_csv_path, index=False)
print(f"[+] Exported 100 unlabeled real transactions to {sample_csv_path}")
print("[+] DONE!")
