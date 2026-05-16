"""call_xai_debug.py — Test /xai_debug endpoint dan submit prediksi baru"""
import urllib.request, json, http.client, uuid, os

BASE = 'http://127.0.0.1:5000'

# ── Step 1: Test /xai_debug route ────────────────────────────────────────────
print("=== Step 1: /xai_debug ===")
try:
    r = urllib.request.urlopen(f'{BASE}/xai_debug', timeout=90)
    data = json.loads(r.read().decode())
    print(f"Tabular XAI : {data.get('tabular')}")
    print(f"Grad-CAM    : {data.get('gradcam')}")
    errs = data.get('errors', [])
    if errs:
        print("ERRORS:")
        for e in errs:
            for k, v in e.items():
                print(f"  [{k}]:\n{v}")
    else:
        print("Errors: none")
    xai_log = data.get('xai_log', '')
    if xai_log:
        print(f"XAI Log:\n{xai_log}")
except Exception as e:
    print(f"xai_debug FAILED: {e}")

# ── Step 2: Submit demo prediction ────────────────────────────────────────────
print("\n=== Step 2: Submit demo prediction ===")
boundary = uuid.uuid4().hex
img_path = r'E:\Diabetic\aptos2019-blindness-detection\train_images\000c1434d8d7.png'

with open(img_path, 'rb') as f:
    img_data = f.read()

fields = {
    'patient_id': 'DEMO-XAI-DEBUG', 'patient_name': 'Test XAI Debug',
    'gender': '0', 'age': '58', 'hypertension': '1', 'heart_disease': '0',
    'smoking_history': '3', 'tinggi_badan': '160', 'berat_badan': '78',
    'hba1c': '8.2', 'blood_glucose': '220',
}

body = b''
for name, value in fields.items():
    body += f'--{boundary}\r\n'.encode()
    body += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
    body += f'{value}\r\n'.encode()
body += f'--{boundary}\r\n'.encode()
body += f'Content-Disposition: form-data; name="fundus_image"; filename="test.png"\r\n'.encode()
body += b'Content-Type: image/png\r\n\r\n'
body += img_data + b'\r\n'
body += f'--{boundary}--\r\n'.encode()

print("Submitting... (tunggu model inference ~30 detik)")
conn = http.client.HTTPConnection('127.0.0.1', 5000, timeout=120)
conn.request('POST', '/predict', body=body,
             headers={'Content-Type': f'multipart/form-data; boundary={boundary}',
                      'Content-Length': str(len(body))})
resp = conn.getresponse()
loc = resp.getheader('Location', '')
print(f"Response: {resp.status}, Location: {loc}")
conn.close()

# ── Step 3: Cek DB record terbaru ────────────────────────────────────────────
print("\n=== Step 3: Cek DB record terbaru ===")
import sqlite3, sys
sys.path.insert(0, 'E:/Diabetic')
db = sqlite3.connect('logs/validation_log.db')
db.row_factory = sqlite3.Row
row = db.execute('SELECT id, gradcam_filename, tabular_explanation_json FROM predictions ORDER BY id DESC LIMIT 1').fetchone()
if row:
    row = dict(row)
    tab = json.loads(row['tabular_explanation_json']) if row['tabular_explanation_json'] else None
    print(f"ID #{row['id']}: Grad-CAM={row['gradcam_filename']}, TabXAI={'OK ('+str(len(tab))+' features)' if tab else 'NONE'}")
db.close()

# ── Step 4: Baca log XAI terbaru ─────────────────────────────────────────────
print("\n=== Step 4: XAI log file ===")
log_path = 'logs/xai_debug.log'
if os.path.exists(log_path):
    with open(log_path, encoding='utf-8') as f:
        content = f.read()
    print(content[-3000:] if content else "(empty)")
else:
    print("Log file belum ada")
