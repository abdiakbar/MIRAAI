"""test_http_routes.py — Verifikasi HTTP routes semua revisi"""
import urllib.request, sqlite3, sys

base = 'http://127.0.0.1:5000'
fails = []

def check(cond, msg):
    if cond:
        print(f'    OK: {msg}')
    else:
        print(f'    FAIL: {msg}')
        fails.append(msg)

print("\n[1/3] Testing Homepage — TB/BB fields...")
r = urllib.request.urlopen(base + '/')
html = r.read().decode()
check(r.status == 200, 'Homepage loads (200)')
check('tinggi_badan' in html, 'Field tinggi_badan ada')
check('berat_badan' in html, 'Field berat_badan ada')
check('name="bmi"' not in html, 'Old BMI name input DIHAPUS')
check('calcBMI' in html, 'Live BMI calculator JS ada')
check('bmi-display' in html, 'BMI display element ada')

print("\n[2/3] Testing History — Delete modal...")
r2 = urllib.request.urlopen(base + '/history')
h2 = r2.read().decode()
check(r2.status == 200, 'History loads (200)')
check('confirmDelete' in h2, 'confirmDelete JS function ada')
check('delete-modal' in h2, 'Modal element ada')
check('Ya, Hapus Sekarang' in h2, 'Tombol konfirmasi hapus ada')
check('btn-danger' in h2, 'Tombol hapus danger-style ada')
check('/delete/' in h2, 'Delete form action URL ada')

print("\n[3/3] Testing Validate — Auto recommendations...")
conn = sqlite3.connect('logs/validation_log.db')
row  = conn.execute('SELECT id FROM predictions ORDER BY id DESC LIMIT 1').fetchone()
conn.close()
if row:
    rid = row[0]
    r3  = urllib.request.urlopen(base + f'/validate/{rid}')
    h3  = r3.read().decode()
    check(r3.status == 200, f'Validate #{rid} loads (200)')
    check('Rekomendasi Otomatis Sistem' in h3, 'Section auto-rekomendasi ada')
    check('name="clinical_preset"' not in h3, 'Checkbox klinis DIHAPUS')
    check('name="nonclinical_preset"' not in h3, 'Checkbox non-klinis DIHAPUS')
    check('clinical_custom' in h3, 'Textarea tambahan klinis ada')
    check('nonclinical_custom' in h3, 'Textarea tambahan non-klinis ada')
    check('doctor_notes' in h3, 'Textarea catatan dokter ada')
    check('doctor_status' in h3, 'Radio validasi status ada')
else:
    print('    SKIP: Tidak ada record di DB untuk test validate')

print()
if fails:
    print(f"HASIL: {len(fails)} CHECK GAGAL:")
    for f in fails: print(f"  - {f}")
    sys.exit(1)
else:
    print("=== SEMUA HTTP ROUTE CHECKS PASSED ===")
