"""check_db_xai.py — Cek status XAI di database terbaru"""
import sqlite3, json, os
os.chdir('E:/Diabetic')

db = sqlite3.connect('logs/validation_log.db')
db.row_factory = sqlite3.Row
rows = db.execute(
    'SELECT id, dr_grade, confidence, gradcam_filename, '
    'tabular_explanation_json, bmi, tinggi_badan, berat_badan '
    'FROM predictions ORDER BY id DESC LIMIT 5'
).fetchall()

print("=== Status XAI di Database (5 record terbaru) ===\n")
for r in rows:
    r = dict(r)
    has_gc  = bool(r.get('gradcam_filename'))
    has_tab = bool(r.get('tabular_explanation_json'))
    tab_cnt = len(json.loads(r['tabular_explanation_json'])) if has_tab else 0
    print(f"ID #{r['id']}:")
    print(f"  Grade={r['dr_grade']}, Confidence={r['confidence']:.1f}%")
    print(f"  TB={r['tinggi_badan']}, BB={r['berat_badan']}, BMI={r['bmi']}")
    print(f"  Grad-CAM  : {'OK -> ' + r['gradcam_filename'] if has_gc else 'TIDAK ADA'}")
    print(f"  Tabular XAI: {'OK -> ' + str(tab_cnt) + ' features' if has_tab else 'TIDAK ADA'}")
    print()

db.close()

# Juga cek file gradcam di uploads/
print("=== File di uploads/ (gradcam) ===")
for f in os.listdir('uploads'):
    if 'gradcam' in f:
        size = os.path.getsize(f'uploads/{f}')
        print(f"  {f} ({size} bytes)")
