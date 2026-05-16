"""
database.py — SQLite helper untuk DR Multi-Modal CDSS
Menyimpan hasil prediksi, validasi dokter, dan data XAI.
"""

import sqlite3
import os
import json
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', 'validation_log.db')


def init_db():
    """Inisialisasi database + migrasi kolom baru (idempoten)."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp               TEXT NOT NULL,
            patient_id              TEXT NOT NULL,
            patient_name            TEXT,
            img_filename            TEXT,
            gender                  INTEGER,
            age                     REAL,
            hypertension            INTEGER,
            heart_disease           INTEGER,
            smoking_history         INTEGER,
            bmi                     REAL,
            hba1c                   REAL,
            blood_glucose           REAL,
            knn_no_diabetes_prob    REAL,
            knn_diabetes_prob       REAL,
            img_probs_json          TEXT,
            fusion_probs_json       TEXT,
            dr_grade                INTEGER,
            dr_label                TEXT,
            confidence              REAL,
            doctor_status           TEXT DEFAULT 'Pending',
            doctor_notes            TEXT,
            clinical_rec_preset     TEXT,
            clinical_rec_custom     TEXT,
            nonclinical_rec_preset  TEXT,
            nonclinical_rec_custom  TEXT,
            rec_timestamp           TEXT
        )
    ''')
    conn.commit()

    # ── Migrasi kolom baru (aman untuk DB existing) ───────────────────────────
    existing = {row[1] for row in c.execute("PRAGMA table_info(predictions)")}
    migrations = [
        ('tinggi_badan',             'REAL'),
        ('berat_badan',              'REAL'),
        ('gradcam_filename',         'TEXT'),   # XAI: path heatmap Grad-CAM
        ('tabular_explanation_json', 'TEXT'),   # XAI: feature risk JSON
    ]
    for col, typ in migrations:
        if col not in existing:
            c.execute(f'ALTER TABLE predictions ADD COLUMN {col} {typ}')
    conn.commit()
    conn.close()


def save_prediction(patient_id, patient_name, img_filename,
                    gender, age, hypertension, heart_disease,
                    smoking_history, bmi, hba1c, blood_glucose,
                    knn_probs, img_probs, fusion_probs,
                    dr_grade, dr_label, confidence,
                    tinggi_badan=None, berat_badan=None,
                    gradcam_filename=None, tabular_explanation=None):
    """Simpan hasil prediksi baru. Return: id record yang baru dibuat."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO predictions (
            timestamp, patient_id, patient_name, img_filename,
            gender, age, hypertension, heart_disease,
            smoking_history, bmi, hba1c, blood_glucose,
            knn_no_diabetes_prob, knn_diabetes_prob,
            img_probs_json, fusion_probs_json,
            dr_grade, dr_label, confidence,
            tinggi_badan, berat_badan,
            gradcam_filename, tabular_explanation_json
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    ''', (
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        patient_id, patient_name or '', img_filename,
        int(gender), float(age), int(hypertension), int(heart_disease),
        int(smoking_history), float(bmi), float(hba1c), float(blood_glucose),
        float(knn_probs[0]), float(knn_probs[1]),
        json.dumps([float(p) for p in img_probs]),
        json.dumps([float(p) for p in fusion_probs]),
        int(dr_grade), str(dr_label), float(confidence),
        float(tinggi_badan) if tinggi_badan is not None else None,
        float(berat_badan)  if berat_badan  is not None else None,
        gradcam_filename,
        json.dumps(tabular_explanation) if tabular_explanation else None,
    ))
    pred_id = c.lastrowid
    conn.commit()
    conn.close()
    return pred_id


def save_validation(pred_id, doctor_status, doctor_notes,
                    clinical_preset, clinical_custom,
                    nonclinical_preset, nonclinical_custom):
    """Update validasi dokter + rekomendasi pada rekaman yang ada."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        UPDATE predictions SET
            doctor_status          = ?,
            doctor_notes           = ?,
            clinical_rec_preset    = ?,
            clinical_rec_custom    = ?,
            nonclinical_rec_preset = ?,
            nonclinical_rec_custom = ?,
            rec_timestamp          = ?
        WHERE id = ?
    ''', (
        doctor_status, doctor_notes,
        clinical_preset, clinical_custom,
        nonclinical_preset, nonclinical_custom,
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        int(pred_id)
    ))
    conn.commit()
    conn.close()


def delete_prediction(pred_id):
    """Hard delete — hapus permanen dari DB. Return img_filename untuk hapus dari disk."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT img_filename, gradcam_filename FROM predictions WHERE id = ?', (int(pred_id),))
    row = c.fetchone()
    img_file    = row[0] if row else None
    gradcam_file = row[1] if row else None
    c.execute('DELETE FROM predictions WHERE id = ?', (int(pred_id),))
    conn.commit()
    conn.close()
    return img_file, gradcam_file   # keduanya dikembalikan ke app.py untuk hapus file


def get_prediction(pred_id):
    """Ambil satu rekaman. Return dict atau None."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM predictions WHERE id = ?', (int(pred_id),))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_predictions(limit=200):
    """Ambil semua rekaman, urut terbaru. Return list of dict."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?', (limit,))
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]
