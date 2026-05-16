"""
submit_demo.py — Submit prediksi demo via HTTP multipart/form-data
Membuat rekaman baru dengan XAI (Grad-CAM + Feature Risk) untuk ditinjau di browser.
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
os.chdir('E:/Diabetic')
sys.path.insert(0, 'E:/Diabetic')

import urllib.request
import urllib.parse
import json

# Kirim via HTTP POST ke Flask (multipart form-data)
import http.client
import mimetypes
import uuid

boundary = uuid.uuid4().hex

def build_multipart(fields, files):
    body = b''
    for name, value in fields.items():
        body += f'--{boundary}\r\n'.encode()
        body += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        body += f'{value}\r\n'.encode()
    for name, (filename, data, ctype) in files.items():
        body += f'--{boundary}\r\n'.encode()
        body += f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode()
        body += f'Content-Type: {ctype}\r\n\r\n'.encode()
        body += data + b'\r\n'
    body += f'--{boundary}--\r\n'.encode()
    return body

# Baca gambar fundus demo
img_path = r'E:\Diabetic\aptos2019-blindness-detection\train_images\000c1434d8d7.png'
with open(img_path, 'rb') as f:
    img_data = f.read()

fields = {
    'patient_id':      'DEMO-XAI-2026',
    'patient_name':    'Pasien Demo XAI',
    'gender':          '0',
    'age':             '58',
    'hypertension':    '1',
    'heart_disease':   '0',
    'smoking_history': '3',
    'tinggi_badan':    '160',
    'berat_badan':     '78',
    'hba1c':           '8.2',
    'blood_glucose':   '220',
}
files = {
    'fundus_image': ('000c1434d8d7.png', img_data, 'image/png')
}

body = build_multipart(fields, files)
content_type = f'multipart/form-data; boundary={boundary}'

conn = http.client.HTTPConnection('127.0.0.1', 5000, timeout=120)
conn.request('POST', '/predict', body=body,
             headers={'Content-Type': content_type,
                      'Content-Length': str(len(body))})

print('[DEMO] Mengirim prediksi ke server... (tunggu ~30 detik untuk model inference + Grad-CAM)')
resp = conn.getresponse()
location = resp.getheader('Location', '')
print(f'[DEMO] Response: {resp.status} {resp.reason}')
print(f'[DEMO] Redirect ke: {location}')

# Ikuti redirect untuk ambil ID prediksi
if resp.status in (301, 302):
    # Extract pred_id dari URL redirect
    parts = location.rstrip('/').split('/')
    pred_id = parts[-1]
    print(f'[DEMO] Rekaman baru ID: #{pred_id}')
    print(f'[DEMO] URL hasil: http://127.0.0.1:5000/result/{pred_id}')

    # Verifikasi XAI data dari DB
    from database import get_prediction
    import json as j
    rec = get_prediction(int(pred_id))
    if rec:
        print(f'[DEMO] Grade DR      : {rec["dr_grade"]} — {rec["dr_label"]}')
        print(f'[DEMO] Confidence    : {rec["confidence"]:.1f}%')
        print(f'[DEMO] BMI (dari TB/BB): {rec["bmi"]:.2f} kg/m²')
        print(f'[DEMO] Grad-CAM      : {rec.get("gradcam_filename") or "TIDAK ADA (gagal)"}')
        tab = rec.get('tabular_explanation_json')
        if tab:
            expl = j.loads(tab)
            print(f'[DEMO] Tabular XAI   : {len(expl)} fitur, top risk = {expl[0]["name"]} ({expl[0]["label"]})')
        else:
            print('[DEMO] Tabular XAI   : TIDAK ADA')
    print(f'\n[DEMO] Buka browser: http://127.0.0.1:5000/result/{pred_id}')
else:
    print(f'[DEMO] Unexpected status: {resp.status}')
conn.close()
