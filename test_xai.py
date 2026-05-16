"""test_xai.py — Verifikasi semua komponen XAI MIRA"""
import os, sys, warnings, json
warnings.filterwarnings('ignore')
os.chdir('E:/Diabetic')
sys.path.insert(0, 'E:/Diabetic')

print("=" * 48)
print("  MIRA XAI Tests — 4 Komponen")
print("=" * 48)

# ── Test 1: explain_tabular ────────────────────────────────
print("\n[1/4] explain_tabular (Feature Risk Contribution)...")
from utils.tabular_handler import explain_tabular, process_tabular
expl = explain_tabular(age=55, hypertension=1, heart_disease=0,
                       bmi=27.6, hba1c=7.8, blood_glucose=195)
assert len(expl) == 6, f"Expected 6, got {len(expl)}"
assert expl[0]['score'] >= expl[-1]['score'], "Not sorted desc"
top = expl[0]
print(f"    OK: {len(expl)} features returned, sorted by risk score")
print(f"    Top risk: {top['name']} = {top['value']} ({top['label']})")

# ── Test 2: generate_gradcam ───────────────────────────────
print("\n[2/4] generate_gradcam (Grad-CAM heatmap)...")
from utils.image_handler import load_efficientnet_b5, generate_gradcam
model, device = load_efficientnet_b5()
img_path  = r'E:\Diabetic\aptos2019-blindness-detection\train_images\000c1434d8d7.png'
save_path = r'E:\Diabetic\uploads\test_gradcam.jpg'
os.makedirs('uploads', exist_ok=True)
result = generate_gradcam(img_path, model, device, target_class=2, save_path=save_path)
assert result is not None, "generate_gradcam returned None"
assert os.path.exists(save_path), "Grad-CAM file not saved"
sz = os.path.getsize(save_path)
print(f"    OK: heatmap saved — {sz} bytes")

# ── Test 3: DB columns ─────────────────────────────────────
print("\n[3/4] Database migration (XAI columns)...")
from database import init_db
init_db()
import sqlite3
conn = sqlite3.connect('logs/validation_log.db')
cols = {r[1] for r in conn.execute("PRAGMA table_info(predictions)")}
conn.close()
for c in ['gradcam_filename', 'tabular_explanation_json', 'tinggi_badan', 'berat_badan']:
    assert c in cols, f"Column {c} missing!"
print(f"    OK: all XAI columns present: {sorted(cols & {'gradcam_filename','tabular_explanation_json'})}")

# ── Test 4: Full save + retrieve ───────────────────────────
print("\n[4/4] Save + retrieve XAI data in DB...")
from utils.fusion import semantic_fusion
from database import save_prediction, get_prediction, delete_prediction
prob_img = [0.10, 0.10, 0.50, 0.20, 0.10]
prob_tab = process_tabular(0, 55, 1, 0, 4, 27.6, 7.8, 195)
fusion   = semantic_fusion(prob_img, prob_tab)
pid = save_prediction(
    patient_id='XAI-TEST', patient_name='Test XAI', img_filename='test.png',
    gender=0, age=55, hypertension=1, heart_disease=0,
    smoking_history=4, bmi=27.6, hba1c=7.8, blood_glucose=195,
    knn_probs=prob_tab, img_probs=prob_img,
    fusion_probs=fusion['combined_probs'],
    dr_grade=fusion['dr_grade'], dr_label=fusion['dr_label'],
    confidence=fusion['confidence'],
    tinggi_badan=165.0, berat_badan=75.0,
    gradcam_filename='test_gradcam.jpg',
    tabular_explanation=expl,
)
rec = get_prediction(pid)
assert rec['gradcam_filename'] == 'test_gradcam.jpg'
parsed = json.loads(rec['tabular_explanation_json'])
assert len(parsed) == 6
delete_prediction(pid)
print(f"    OK: record #{pid} saved, XAI data persisted + retrieved correctly")

print()
print("=" * 48)
print("  SEMUA XAI TESTS PASSED (4/4)")
print("=" * 48)
