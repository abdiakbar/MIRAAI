"""
utils/tabular_handler.py — KNN inference + Explainable AI (Feature Risk Contribution)

scaler_knn.pkl dilatih dengan 6 fitur:
  [age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level]
Gender & smoking_history dikumpulkan untuk rekam medis, TIDAK dimasukkan ke KNN.
"""

import joblib
import numpy as np
import os

_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')

scaler    = joblib.load(os.path.join(_MODEL_DIR, 'scaler_knn.pkl'))
model_knn = joblib.load(os.path.join(_MODEL_DIR, 'model_knn.pkl'))

GENDER_ENCODE  = {'Female': 0, 'Male': 1, 'Other': 2}
SMOKING_ENCODE = {'No Info': 0, 'current': 1, 'ever': 2,
                  'former': 3, 'never': 4, 'not current': 5}


# ─── KNN Inference ────────────────────────────────────────────────────────────

def process_tabular(gender, age, hypertension, heart_disease,
                    smoking_history, bmi, hba1c, blood_glucose):
    """
    Return [P(no_diabetes), P(diabetes)] dari KNN.
    Urutan fitur sesuai scaler.feature_names_in_:
    ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    """
    features = np.array([[
        age, int(hypertension), int(heart_disease), bmi, hba1c, blood_glucose
    ]], dtype=np.float64)
    scaled = scaler.transform(features)
    return model_knn.predict_proba(scaled)[0]


# ─── XAI: Feature Risk Contribution ──────────────────────────────────────────

# Ambang klinis berbasis panduan ADA (American Diabetes Association)
_RANGES = {
    'hba1c': {
        'normal':  (0,    5.7,  'Normal',      '< 5.7%'),
        'warning': (5.7,  6.5,  'Pre-Diabetes', '5.7 – 6.4%'),
        'danger':  (6.5,  99,   'Diabetes',     '≥ 6.5%'),
    },
    'blood_glucose': {
        'normal':  (0,    100,  'Normal',       '70 – 99 mg/dL'),
        'warning': (100,  126,  'Pre-Diabetes', '100 – 125 mg/dL'),
        'danger':  (126,  9999, 'Diabetes',     '≥ 126 mg/dL'),
    },
    'bmi': {
        'normal':  (0,    25,   'Normal',      '18.5 – 24.9'),
        'warning': (25,   30,   'Overweight',  '25.0 – 29.9'),
        'danger':  (30,   99,   'Obesitas',    '≥ 30.0'),
    },
    'age': {
        'normal':  (0,    45,   'Risiko Rendah',   '< 45 tahun'),
        'warning': (45,   60,   'Risiko Sedang',   '45 – 59 tahun'),
        'danger':  (60,   150,  'Risiko Tinggi',   '≥ 60 tahun'),
    },
}


def _classify(key, value):
    """Return (status, label, range_label, score 0–1) untuk satu fitur numerik."""
    for status in ('danger', 'warning', 'normal'):
        lo, hi, label, range_lbl = _RANGES[key][status]
        if lo <= value < hi:
            # Hitung skor kontribusi risiko (0 = aman, 1 = sangat berisiko)
            if status == 'danger':
                score = 0.75 + min((value - lo) / max(hi - lo, 1), 0.25)
            elif status == 'warning':
                score = 0.40 + min((value - lo) / max(hi - lo, 1), 0.35)
            else:
                score = max(0.05, (value - lo) / max(hi - lo, 1) * 0.35)
            return status, label, range_lbl, round(min(score, 1.0), 3)
    return 'normal', 'Normal', '—', 0.05


def explain_tabular(age, hypertension, heart_disease, bmi, hba1c, blood_glucose):
    """
    Analisis kontribusi risiko tiap fitur klinis terhadap prediksi KNN.

    Returns:
        List of dict — diurutkan dari kontribusi risiko TERTINGGI ke terendah:
        [
          {
            'name'       : str   — nama fitur,
            'value'      : str   — nilai aktual dengan satuan,
            'status'     : str   — 'normal' | 'warning' | 'danger',
            'label'      : str   — interpretasi klinis,
            'range_label': str   — rentang referensi,
            'score'      : float — skor risiko 0.0–1.0 (untuk panjang bar),
            'pct'        : int   — score × 100 (untuk CSS width),
          }
        ]
    """
    features = []

    # ── Fitur numerik ──────────────────────────────────────────────
    for key, val, name, unit in [
        ('hba1c',         hba1c,         'HbA1c Level',   '%'),
        ('blood_glucose', blood_glucose, 'Blood Glucose', 'mg/dL'),
        ('bmi',           bmi,           'BMI',           'kg/m²'),
        ('age',           age,           'Usia',          'tahun'),
    ]:
        status, label, range_lbl, score = _classify(key, val)
        val_str = (f'{val:.1f} {unit}' if isinstance(val, float) else f'{int(val)} {unit}')
        features.append({
            'name': name, 'value': val_str, 'status': status,
            'label': label, 'range_label': range_lbl,
            'score': score, 'pct': int(score * 100),
        })

    # ── Fitur biner ────────────────────────────────────────────────
    for name, flag in [('Hipertensi', hypertension), ('Penyakit Jantung', heart_disease)]:
        if int(flag):
            features.append({
                'name': name, 'value': 'Ya', 'status': 'danger',
                'label': 'Faktor Risiko Aktif', 'range_label': 'Meningkatkan risiko DR',
                'score': 0.72, 'pct': 72,
            })
        else:
            features.append({
                'name': name, 'value': 'Tidak', 'status': 'normal',
                'label': 'Tidak Ada Risiko', 'range_label': 'Tidak meningkatkan risiko',
                'score': 0.08, 'pct': 8,
            })

    # Urutkan dari risiko tertinggi ke terendah
    features.sort(key=lambda x: x['score'], reverse=True)
    return features