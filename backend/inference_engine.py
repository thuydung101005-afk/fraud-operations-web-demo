import os
import time
import json
import joblib
import numpy as np
import pandas as pd

class FraudInferenceEngine:
    def __init__(self, models_dir=None):
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(__file__), "models")
        
        self.models_dir = models_dir
        self.model = joblib.load(os.path.join(models_dir, "master_model.joblib"))
        self.scaler = joblib.load(os.path.join(models_dir, "scaler.joblib"))
        self.meta = joblib.load(os.path.join(models_dir, "meta_features.joblib"))
        
        self.features = self.meta['features']
        self.threshold = self.meta['threshold'] # 0.0227 (2.27%)
        self.critical_threshold = self.meta.get('critical_threshold', 0.15) # 15.0%
        self.global_median_amt = self.meta['global_median_amt']
        self.global_fraud_rate = self.meta['global_fraud_rate']
        self.cat_fraud_rates = self.meta['cat_fraud_rates']
        self.card_hist_mean = self.meta['card_hist_mean']
        self.card_hist_count = self.meta['card_hist_count']

    @staticmethod
    def haversine_np(lat1, lon1, lat2, lon2):
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return 6371.0 * c

    def preprocess_and_extract_features(self, df_raw):
        df = df_raw.copy()
        
        # Ensure datetimes
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
        df['dob'] = pd.to_datetime(df['dob'])
        
        # Sort chronologically if needed
        df = df.sort_values('trans_date_trans_time').reset_index(drop=True)
        
        # 1. Distance & Age
        df['dist_to_merchant_km'] = self.haversine_np(df['lat'], df['long'], df['merch_lat'], df['merch_long'])
        df['age'] = df['trans_date_trans_time'].dt.year - df['dob'].dt.year
        
        # 2. Time cyclical features
        hour = df['trans_date_trans_time'].dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
        df['is_night'] = hour.between(0, 5).astype(int)
        
        # 3. Card history profile lookups + batch expanding
        def get_past_mean(row):
            cc_str = str(row['cc_num'])
            if cc_str in self.card_hist_mean:
                return self.card_hist_mean[cc_str]
            return self.global_median_amt

        def get_past_count(row):
            cc_str = str(row['cc_num'])
            if cc_str in self.card_hist_count:
                return self.card_hist_count[cc_str]
            return 0

        past_means = df.apply(get_past_mean, axis=1)
        past_counts = df.apply(get_past_count, axis=1)

        # In-batch expanding update if multiple transactions for same card
        if df['cc_num'].duplicated().any():
            batch_exp_mean = df.groupby('cc_num')['amt'].transform(lambda s: s.shift(1).expanding().mean())
            past_means = batch_exp_mean.fillna(past_means)
            batch_exp_count = df.groupby('cc_num')['amt'].transform(lambda s: s.shift(1).expanding().count())
            past_counts = past_counts + batch_exp_count.fillna(0)

        past_means = np.where(past_means <= 0, self.global_median_amt, past_means)
        
        df['amt_ratio'] = df['amt'] / past_means
        df['amt_diff'] = df['amt'] - past_means
        df['trans_count_past'] = past_counts
        
        # 4. Category Risk
        df['category_risk'] = df['category'].map(self.cat_fraud_rates).fillna(self.global_fraud_rate)
        
        return df

    def predict_batch(self, df_raw):
        t0 = time.perf_counter()
        df_feat = self.preprocess_and_extract_features(df_raw)
        
        X = df_feat[self.features]
        X_scaled = self.scaler.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.features)
        
        # Model inference using exact LightGBM Master Model
        probs = self.model.predict_proba(X_scaled_df)[:, 1]
        latency_ms = (time.perf_counter() - t0) * 1000
        avg_latency_per_tx = latency_ms / max(len(df_raw), 1)
        
        results = []
        for i in range(len(df_feat)):
            row = df_feat.iloc[i]
            prob = float(probs[i])
            risk_pct = round(prob * 100, 2)
            
            # Decision Policy
            if prob < self.threshold:
                status = "Đã duyệt"
                decision_badge = "AUTO-APPROVED"
                color = "green"
            elif prob <= self.critical_threshold:
                status = "Cần gọi CSKH"
                decision_badge = "STEP-UP AUTH"
                color = "amber"
            else:
                status = "Đã chặn"
                decision_badge = "AUTO-BLOCKED"
                color = "red"
                
            # Explainable AI (XAI) insights
            insights = []
            if row['amt_ratio'] >= 3.0:
                insights.append({
                    'type': 'red',
                    'title': 'Số tiền chi tiêu đột biến',
                    'desc': f"Số tiền ${row['amt']:,.2f} gấp {row['amt_ratio']:.1f}x so với mức chi tiêu trung bình (${row['amt'] - row['amt_diff']:,.2f}) của chủ thẻ."
                })
            elif row['amt'] > 300:
                insights.append({
                    'type': 'amber',
                    'title': 'Giá trị giao dịch lớn',
                    'desc': f"Giao dịch ${row['amt']:,.2f} vượt ngưỡng giao dịch thông thường."
                })
                
            if row['is_night'] == 1:
                insights.append({
                    'type': 'amber',
                    'title': 'Khung giờ rủi ro cao (Đêm khuya)',
                    'desc': f"Giao dịch được thực hiện lúc {row['trans_date_trans_time'].strftime('%H:%M:%S')} (khung giờ 00:00 - 05:00 có tần suất gian lận cao)."
                })
                
            if row['dist_to_merchant_km'] >= 100:
                insights.append({
                    'type': 'red' if row['dist_to_merchant_km'] > 500 else 'amber',
                    'title': 'Bất thường khoảng cách địa lý',
                    'desc': f"Máy quẹt thẻ / Merchant cách nơi cư trú của chủ thẻ {row['dist_to_merchant_km']:.1f} km."
                })
                
            cat_risk_val = float(row['category_risk'])
            if cat_risk_val >= self.global_fraud_rate * 2.0:
                insights.append({
                    'type': 'amber',
                    'title': f"Ngành hàng rủi ro cao ({row['category']})",
                    'desc': f"Ngành hàng này có tỷ lệ gian lận lịch sử {cat_risk_val*100:.2f}% (cao gấp {cat_risk_val/self.global_fraud_rate:.1f}x mức trung bình toàn hệ thống)."
                })

            if not insights:
                insights.append({
                    'type': 'green',
                    'title': 'Hồ sơ giao dịch an toàn',
                    'desc': 'Số tiền, vị trí và thời gian đều khớp với thói quen sinh hoạt và lịch sử thẻ của khách hàng.'
                })

            # Masked card
            cc_str = str(row['cc_num'])
            masked_card = f"**** {cc_str[-4:]}" if len(cc_str) >= 4 else f"**** {cc_str}"
            
            # Format display item
            results.append({
                'id': str(row.get('trans_num', f"TX-{i+1:06d}")),
                'date': row['trans_date_trans_time'].strftime('%d %b %Y'),
                'time': row['trans_date_trans_time'].strftime('%H:%M:%S'),
                'name': f"{row.get('first', '')} {row.get('last', '')}".strip() or "Khách hàng Thẻ",
                'card': masked_card,
                'raw_cc_num': cc_str,
                'amount': float(row['amt']),
                'category': str(row['category']),
                'merchant': str(row.get('merchant', 'Merchant')).replace('fraud_', ''),
                'risk': risk_pct,
                'risk_proba': prob,
                'status': status,
                'decision_badge': decision_badge,
                'color': color,
                'age': int(row['age']),
                'gender': str(row.get('gender', 'N/A')),
                'city': str(row.get('city', '')),
                'state': str(row.get('state', '')),
                'address': f"{row.get('city', '')}, {row.get('state', '')}".strip(', '),
                'job': str(row.get('job', 'N/A')),
                'location': f"Lat: {row['merch_lat']:.2f}, Long: {row['merch_long']:.2f}",
                'distance': round(float(row['dist_to_merchant_km']), 1),
                'amt_ratio': round(float(row['amt_ratio']), 1),
                'category_risk': round(cat_risk_val * 100, 2),
                'insights': insights
            })
            
        return {
            'transactions': results,
            'summary': {
                'total': len(results),
                'approved': sum(1 for r in results if r['status'] == 'Đã duyệt'),
                'cskh_review': sum(1 for r in results if r['status'] == 'Cần gọi CSKH'),
                'blocked': sum(1 for r in results if r['status'] == 'Đã chặn'),
                'total_amount': sum(r['amount'] for r in results),
                'avg_latency_ms': round(avg_latency_per_tx, 4),
                'total_latency_ms': round(latency_ms, 2),
                'threshold': self.threshold
            }
        }

if __name__ == "__main__":
    engine = FraudInferenceEngine()
    demo_csv = os.path.join(os.path.dirname(__file__), "..", "fraud-portal", "unlabeled_transactions_demo.csv")
    df = pd.read_csv(demo_csv)
    print(f"[*] Testing scoring on {len(df)} unlabeled transactions...")
    res = engine.predict_batch(df)
    print(f"[+] Total scored   : {res['summary']['total']}")
    print(f"[+] Approved       : {res['summary']['approved']}")
    print(f"[+] CSKH Review    : {res['summary']['cskh_review']}")
    print(f"[+] Blocked        : {res['summary']['blocked']}")
    print(f"[+] Avg Latency    : {res['summary']['avg_latency_ms']:.4f} ms / tx")
    print(f"[+] Total Latency  : {res['summary']['total_latency_ms']:.2f} ms")
