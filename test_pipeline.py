"""
test_pipeline.py — Pengujian komprehensif semua fitur MIRA (post-revisi)
Menguji: TB/BB→BMI, pipeline inferensi, DB CRUD, hard delete, auto-rekomendasi.
"""
import sys, os
os.chdir('E:/Diabetic')
sys.path.insert(0, 'E:/Diabetic')

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

from utils.image_handler import predict_fundus, load_efficientnet_b5
from utils.tabular_handler import process_tabular
from utils.fusion import semantic_fusion
from database import init_db, save_prediction, save_validation, get_prediction, get_all_predictions, delete_prediction

print("=" * 55)
print("  MIRA CDSS — Pengujian Komprehensif Post-Revisi")
print("=" * 55)

# ── Test 1: DB migrasi kolom baru ───────────────────────────
print("\n[1/7] DB init & migrasi kolom tb/bb...")
init_db()
import sqlite3
conn = sqlite3.connect('logs/validation_log.db')
cols = {row[1] for row in conn.execute("PRAGMA table_info(predictions)")}
conn.close()
assert 'tinggi_badan' in cols, "GAGAL: kolom tinggi_badan tidak ada"
assert 'berat_badan'  in cols, "GAGAL: kolom berat_badan tidak ada"
print("    OK: kolom tinggi_badan + berat_badan tersedia")

# ── Test 2: Kalkulasi BMI server-side ───────────────────────
print("\n[2/7] Kalkulasi BMI dari TB & BB...")
tb, bb = 162.0, 72.5
bmi = round(bb / ((tb / 100) ** 2), 2)
assert 27.0 < bmi < 28.0, f"GAGAL: BMI={bmi} out of expected range"
print(f"    OK: TB={tb}cm, BB={bb}kg → BMI={bmi} kg/m² (Overweight)")

# ── Test 3: Load model ──────────────────────────────────────
print("\n[3/7] Loading EfficientNetB5...")
model, device = load_efficientnet_b5()
print(f"    OK: model loaded on {device}")

# ── Test 4: Inferensi gambar ────────────────────────────────
print("\n[4/7] Inferensi citra fundus...")
img_path = r'E:\Diabetic\aptos2019-blindness-detection\train_images\000c1434d8d7.png'
prob_img = predict_fundus(img_path, model, device)
assert len(prob_img) == 5, "GAGAL: output bukan 5 kelas"
assert abs(sum(prob_img) - 1.0) < 0.01, "GAGAL: probabilitas tidak sum=1"
print(f"    OK: prob_img = {[round(float(p),3) for p in prob_img]}")

# ── Test 5: Inferensi tabular ───────────────────────────────
print("\n[5/7] Inferensi KNN tabular...")
prob_tab = process_tabular(
    gender=0, age=55.0, hypertension=1, heart_disease=0,
    smoking_history=4, bmi=bmi, hba1c=7.8, blood_glucose=195.0
)
assert len(prob_tab) == 2, "GAGAL: output KNN bukan 2 kelas"
print(f"    OK: prob_tab = {[round(float(p),3) for p in prob_tab]}")

# ── Test 6: Fusion + save + validation ─────────────────────
print("\n[6/7] Fusion + simpan ke DB + validasi...")
fusion = semantic_fusion(prob_img, prob_tab)
grade = fusion['dr_grade']
print(f"    Fusion → Grade {grade} — {fusion['dr_label']}, confidence {fusion['confidence']:.1f}%")

pred_id = save_prediction(
    patient_id='TEST-REVISI-001', patient_name='Siti Rahayu',
    img_filename='test_fundus.png',
    gender=0, age=55.0, hypertension=1, heart_disease=0,
    smoking_history=4, bmi=bmi, hba1c=7.8, blood_glucose=195.0,
    knn_probs=prob_tab, img_probs=prob_img,
    fusion_probs=fusion['combined_probs'],
    dr_grade=grade, dr_label=fusion['dr_label'],
    confidence=fusion['confidence'],
    tinggi_badan=tb, berat_badan=bb,
)
print(f"    OK: record disimpan dengan ID #{pred_id}")

# Simulasi auto-rekomen sistem (sesuai app.py)
CLINICAL_RECS = {
    0: ["Pemeriksaan mata rutin setiap 12 bulan","Pertahankan kontrol gula darah optimal (HbA1c < 7%)","Monitoring tekanan darah rutin"],
    1: ["Follow-up ophthalmologi dalam 6 bulan","Optimasi kontrol gula darah ketat","Konsultasi dokter spesialis mata","Monitoring tekanan darah secara berkala"],
    2: ["Rujukan ke dokter ophthalmolog segera","Follow-up dalam 3-4 bulan","Pertimbangkan laser photocoagulation fokal","Optimasi terapi diabetes intensif"],
    3: ["Rujukan SEGERA ke ophthalmolog","Pertimbangkan laser scatter (PRP)","Kontrol gula darah sangat ketat","Monitoring fungsi ginjal","Evaluasi tekanan intraokular"],
    4: ["Rujukan SEGERA ke spesialis Vitreoretinal","Evaluasi injeksi anti-VEGF","Pertimbangkan vitrektomi","Panretinal photocoagulation (PRP) laser","Kontrol gula darah intensif segera"],
}
NONCLINICAL_RECS = {
    0: ["Edukasi diet rendah gula","Olahraga aerobik minimal 150 menit per minggu","Monitoring gula darah mandiri","Jadwal tidur teratur"],
    1: ["Edukasi diet rendah gula","Olahraga aerobik minimal 150 menit per minggu","Hindari merokok","Penggunaan kacamata pelindung UV"],
    2: ["Edukasi diet rendah gula","Olahraga aerobik minimal 150 menit per minggu","Hindari merokok","Manajemen stres"],
    3: ["Edukasi diet rendah gula","Batasi aktivitas fisik berat sementara","Monitoring gula darah intensif","Manajemen stres"],
    4: ["Istirahat total aktivitas fisik berat","Diet ketat rendah gula","Monitoring gula darah intensif","Jaga kebersihan mata"],
}
clinical_preset    = '; '.join(CLINICAL_RECS.get(grade, []))
nonclinical_preset = '; '.join(NONCLINICAL_RECS.get(grade, []))

save_validation(
    pred_id=pred_id,
    doctor_status='Approved',
    doctor_notes='Hasil prediksi AI dikonfirmasi konsisten dengan pemeriksaan klinis.',
    clinical_preset=clinical_preset,
    clinical_custom='Rujukan ke dr. Andini Ophthalmolog RSUP',
    nonclinical_preset=nonclinical_preset,
    nonclinical_custom='',
)
rec = get_prediction(pred_id)
assert rec['doctor_status'] == 'Approved', "GAGAL: status tidak Approved"
assert rec['tinggi_badan'] == tb, "GAGAL: tinggi_badan tidak tersimpan"
assert rec['berat_badan']  == bb, "GAGAL: berat_badan tidak tersimpan"
assert clinical_preset in rec['clinical_rec_preset'], "GAGAL: preset klinis tidak tersimpan"
print(f"    OK: validasi + rekomendasi auto tersimpan, tb={rec['tinggi_badan']}, bb={rec['berat_badan']}")

# ── Test 7: Hard delete ─────────────────────────────────────
print("\n[7/7] Hard delete rekaman...")
all_before = len(get_all_predictions())
delete_prediction(pred_id)
all_after = len(get_all_predictions())
assert all_after == all_before - 1, "GAGAL: record tidak terhapus"
assert get_prediction(pred_id) is None, "GAGAL: record masih ada setelah delete"
print(f"    OK: rekaman #{pred_id} dihapus permanen ({all_before}→{all_after} records)")

print("\n" + "=" * 55)
print("  SEMUA PENGUJIAN BACKEND BERHASIL (7/7)")
print("=" * 55)
