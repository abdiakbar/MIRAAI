# MIRA — Multimodal Intelligent Retinal Assistant
## Clinical Decision Support System untuk Diabetic Retinopathy

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.x-green)](https://flask.palletsprojects.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org)
[![SQLite](https://img.shields.io/badge/Database-SQLite-lightgrey)](https://sqlite.org)

---

## Daftar Isi
1. [Gambaran Umum Sistem](#1-gambaran-umum-sistem)
2. [Arsitektur Pipeline](#2-arsitektur-pipeline)
3. [Struktur Folder](#3-struktur-folder)
4. [Instalasi & Setup](#4-instalasi--setup)
5. [Menjalankan Aplikasi](#5-menjalankan-aplikasi)
6. [Panduan Penggunaan](#6-panduan-penggunaan)
7. [Fitur Explainable AI (XAI)](#7-fitur-explainable-ai-xai)
8. [Database SQLite](#8-database-sqlite)
9. [Melakukan Perubahan Lebih Lanjut](#9-melakukan-perubahan-lebih-lanjut)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Gambaran Umum Sistem

MIRA adalah aplikasi web **Clinical Decision Support System (CDSS)** yang mengklasifikasikan tingkat keparahan **Diabetic Retinopathy (DR)** menggunakan pendekatan multimodal:

| Modalitas | Model | Input |
|-----------|-------|-------|
| **Citra Fundus** | EfficientNetB5 (PyTorch) | Foto funduskopi retina |
| **Data Klinis** | K-Nearest Neighbors (scikit-learn) | 6 fitur numerik klinis |
| **Fusion** | Semantic Probabilistic Projection | Gabungan kedua model |

**Output:** Prediksi Grade DR (0–4) dengan confidence score + **Penjelasan AI (XAI)**

**Kelas Prediksi:**
| Grade | Label | Deskripsi |
|-------|-------|-----------|
| 0 | No DR | Tidak ada retinopathy |
| 1 | Mild DR | Retinopathy ringan |
| 2 | Moderate DR | Retinopathy sedang |
| 3 | Severe DR | Retinopathy berat |
| 4 | Proliferative DR | Retinopathy proliferatif |

---

## 2. Arsitektur Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                      INPUT PASIEN                           │
│   Foto Fundus  +  TB/BB/HbA1c/Glukosa/Hipertensi/dll       │
└────────────┬────────────────────────┬───────────────────────┘
             │                        │
     ┌───────▼────────┐      ┌────────▼────────┐
     │  IMAGE PIPELINE │      │ TABULAR PIPELINE │
     │                │      │                  │
     │ Circular Crop  │      │ StandardScaler   │
     │ CLAHE (LAB)    │      │ (scaler_knn.pkl) │
     │ 128×128 resize │      │                  │
     │ /255 normalize │      │ KNN Classifier   │
     │                │      │ (model_knn.pkl)  │
     │ EfficientNetB5 │      │                  │
     │ (best_model.pth│      │ P(no_DM), P(DM)  │
     │ 5-class softmax│      │                  │
     └───────┬────────┘      └────────┬─────────┘
             │                        │
     ┌───────▼────────────────────────▼─────────┐
     │          LATE FUSION (70% / 30%)          │
     │  Semantic Probabilistic Projection        │
     │  P(DM) dari KNN → distribusi Grade 1–4   │
     └───────────────────┬───────────────────────┘
                         │
     ┌───────────────────▼───────────────────────┐
     │              OUTPUT + XAI                  │
     │  Grade DR + Confidence + Grad-CAM heatmap  │
     │  + Feature Risk Contribution Chart         │
     └───────────────────────────────────────────┘
```

### Bobot Fusion
- **EfficientNetB5 (Citra):** 70%
- **KNN (Tabular):** 30%

### Fitur KNN (6 fitur)
```
age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level
```
> Gender & smoking_history dikumpulkan untuk rekam medis, **tidak dimasukkan ke KNN** (sesuai training scaler).

### Kalkulasi BMI
```
BMI = berat_badan (kg) / (tinggi_badan (m))²
```
Dihitung otomatis dari input TB (cm) dan BB (kg).

---

## 3. Struktur Folder

```
E:\Diabetic\
│
├── app.py                      # Flask application utama
├── database.py                 # SQLite helper (CRUD + migrasi)
├── requirements.txt            # Dependencies Python
├── README.md                   # Dokumentasi ini
│
├── models/                     # File model terlatih
│   ├── baseline_efficientnetb5_model_V2.pth   # Bobot CNN
│   ├── model_knn.pkl                          # Model KNN
│   └── scaler_knn.pkl                         # StandardScaler KNN
│
├── utils/
│   ├── image_handler.py        # Pipeline citra + Grad-CAM XAI
│   ├── tabular_handler.py      # Pipeline KNN + Feature Risk XAI
│   └── fusion.py               # Late fusion logic
│
├── templates/
│   ├── base.html               # Layout dasar (navbar, flash)
│   ├── index.html              # Form input pasien
│   ├── result.html             # Halaman hasil prediksi + XAI
│   ├── validate.html           # Form validasi dokter
│   └── history.html            # Riwayat semua prediksi
│
├── static/
│   ├── css/style.css           # Green Glassmorphism theme
│   └── js/main.js              # JavaScript utilities
│
├── uploads/                    # Gambar fundus + Grad-CAM heatmap
│   ├── <uuid>.png              # Gambar fundus asli
│   └── <uuid>_gradcam.jpg      # Heatmap Grad-CAM overlay
│
├── logs/
│   └── validation_log.db       # Database SQLite
│
└── test_pipeline.py            # Script pengujian backend
    test_xai.py                 # Script pengujian XAI
    test_http_routes.py         # Script pengujian HTTP routes
```

---

## 4. Instalasi & Setup

### Prasyarat
- Python 3.9 atau lebih baru
- pip

### Langkah Instalasi

```powershell
# 1. Masuk ke direktori proyek
cd E:\Diabetic

# 2. (Opsional) Buat virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

# 3. Install semua dependencies
pip install -r requirements.txt
```

### Verifikasi File Model

Pastikan ketiga file model ini ada:
```
models/baseline_efficientnetb5_model_V2.pth   (~100–200 MB)
models/model_knn.pkl
models/scaler_knn.pkl
```

### Inisialisasi Database

Database dibuat **otomatis** saat aplikasi pertama kali dijalankan. Tidak perlu langkah manual.

---

## 5. Menjalankan Aplikasi

```powershell
cd E:\Diabetic
python app.py
```

Output yang diharapkan:
```
[DR-CDSS] Loading EfficientNetB5 model...
[DR-CDSS] Model ready on: cpu
 * Running on http://127.0.0.1:5000
```

Buka browser dan akses: **http://127.0.0.1:5000**

> **Catatan:** `use_reloader=False` diatur agar model tidak dimuat dua kali saat debug mode.

### Menghentikan Server
Tekan `Ctrl + C` di terminal.

---

## 6. Panduan Penggunaan

### 6.1 Form Input Pasien (`/`)

| Field | Keterangan | Wajib |
|-------|-----------|-------|
| ID Pasien | Format bebas (misal: RM-2024-001) | ✅ Ya |
| Nama Pasien | Nama lengkap | Tidak |
| Jenis Kelamin | Perempuan / Laki-laki / Lainnya | ✅ Ya |
| Usia | Dalam tahun | ✅ Ya |
| Hipertensi | Ya / Tidak | ✅ Ya |
| Penyakit Jantung | Ya / Tidak | ✅ Ya |
| Riwayat Merokok | Dropdown 6 pilihan | ✅ Ya |
| **Tinggi Badan (cm)** | Input manual | ✅ Ya |
| **Berat Badan (kg)** | Input manual | ✅ Ya |
| HbA1c Level (%) | Range 3.5–9.0 | ✅ Ya |
| Blood Glucose (mg/dL) | Range 80–300 | ✅ Ya |
| Citra Fundus | PNG/JPG/TIFF, max 16MB | ✅ Ya |

> BMI dihitung otomatis oleh sistem dari TB dan BB.

### 6.2 Halaman Hasil (`/result/<id>`)

Menampilkan:
- **Badge Grade DR** (0–4) dengan confidence ring
- **Distribusi probabilitas** 5 kelas (bar chart)
- **Breakdown model** — EfficientNetB5 (70%) vs KNN (30%)
- **🔍 Penjelasan AI (XAI):**
  - Grad-CAM heatmap (fundus asli vs overlay)
  - Kontribusi faktor risiko klinis (bar chart per fitur)
- **Data klinis pasien** (TB, BB, BMI terhitung, dll)

### 6.3 Validasi Dokter (`/validate/<id>`)

Dokter melakukan:
1. **Pilih status:** ✅ Approved / ❌ Rejected / ⚠️ Needs Review
2. **Tulis catatan** klinis (opsional)
3. **Tambah rekomendasi kustom** klinis & non-klinis (opsional)

> Rekomendasi klinis dan non-klinis **di-generate otomatis oleh sistem** berdasarkan Grade DR — dokter hanya perlu menambahkan catatan tambahan jika diperlukan.

### 6.4 Riwayat Prediksi (`/history`)

- **Filter** berdasarkan pencarian teks atau status validasi
- **Lihat detail** prediksi (tombol 👁️)
- **Edit validasi** (tombol ✏️)
- **Hapus permanen** (tombol 🗑️ + modal konfirmasi)

---

## 7. Aksesibilitas & Standar UI (v2.0)

MIRA v2.0 telah ditingkatkan untuk memenuhi standar **WCAG 2.1 AA** dan prinsip **Heuristic Evaluation**:

### 7.1 Kepatuhan Aksesibilitas
- **Semantic HTML5:** Menggunakan landmark `<nav>`, `<main>`, `<header>`, `<footer>`, dan `fieldset`/`legend`.
- **Keyboard Navigation:** Focus ring yang visible (`:focus-visible`) dan navigasi logis.
- **Screen Reader Support:** Penambahan `aria-label`, `aria-describedby`, `aria-current`, dan `sr-only` text.
- **Skip Link:** Navigasi cepat ke konten utama bagi pengguna keyboard/pembaca layar.
- **Contrast Ratio:** Palet warna Emerald telah diuji untuk rasio kontras minimal 4.5:1.

### 7.2 Alur Validasi Klinis
Sistem ini memfasilitasi workflow dokter melalui:
1. **Visual Hierarchy:** Badge grade DR yang mencolok dan confidence ring untuk atensi cepat.
2. **Status Card Interface:** Penggantian radio button standar dengan kartu status visual yang intuitif (Approved/Rejected/Needs Review).
3. **Sticky CTA Footer:** Tombol aksi validasi selalu terlihat di layar dokter saat melakukan review panjang.
4. **Explainable AI:** Transparansi model citra (Grad-CAM) dan fitur klinis untuk mendukung diagnosis objektif.

---

## 8. Fitur Explainable AI (XAI)

### 7.1 Grad-CAM (Citra Fundus)

**Cara kerja:**
1. Hook dipasang pada `model.features[-1]` (layer terakhir EfficientNetB5)
2. Forward pass dilakukan dengan gradient aktif (`torch.enable_grad()`)
3. Backward pass dari skor kelas prediksi
4. Gradien dirata-rata per channel → bobot aktivasi
5. Weighted sum activation map → heatmap → overlay JET colormap

**Interpretasi warna:**
| Warna | Makna |
|-------|-------|
| 🔴 Merah / Kuning | Area sangat berpengaruh terhadap prediksi |
| 🟢 Hijau | Kontribusi sedang |
| 🔵 Biru | Kontribusi rendah |

**File disimpan:** `uploads/<uuid>_gradcam.jpg`

### 7.2 Feature Risk Contribution (KNN Tabular)

Membandingkan nilai fitur pasien terhadap **ambang klinis ADA (American Diabetes Association)**:

| Fitur | Normal | Pre-Diabetes | Diabetes/Risiko |
|-------|--------|--------------|-----------------|
| HbA1c | < 5.7% | 5.7–6.4% | ≥ 6.5% |
| Blood Glucose | 70–99 mg/dL | 100–125 mg/dL | ≥ 126 mg/dL |
| BMI | 18.5–24.9 | 25.0–29.9 | ≥ 30.0 |
| Usia | < 45 th | 45–59 th | ≥ 60 th |
| Hipertensi | Tidak | — | Ya |
| Penyakit Jantung | Tidak | — | Ya |

**Hasil ditampilkan sebagai bar chart** dengan kode warna:
- 🟢 Hijau = Normal
- 🟡 Kuning = Pre-Diabetes / Batas
- 🔴 Merah = Diabetes / Risiko Tinggi

---

## 8. Database SQLite

### Lokasi
```
logs/validation_log.db
```

### Skema Tabel `predictions`

| Kolom | Tipe | Keterangan |
|-------|------|-----------|
| id | INTEGER | Primary key |
| timestamp | TEXT | Waktu prediksi |
| patient_id | TEXT | ID pasien unik |
| patient_name | TEXT | Nama pasien |
| img_filename | TEXT | Nama file gambar fundus |
| gender | INTEGER | 0=F, 1=M, 2=Other |
| age | REAL | Usia dalam tahun |
| hypertension | INTEGER | 0/1 |
| heart_disease | INTEGER | 0/1 |
| smoking_history | INTEGER | 0–5 |
| bmi | REAL | BMI terhitung |
| tinggi_badan | REAL | TB dalam cm |
| berat_badan | REAL | BB dalam kg |
| hba1c | REAL | HbA1c level % |
| blood_glucose | REAL | Gula darah mg/dL |
| knn_no_diabetes_prob | REAL | P(no diabetes) dari KNN |
| knn_diabetes_prob | REAL | P(diabetes) dari KNN |
| img_probs_json | TEXT | JSON array 5-kelas dari CNN |
| fusion_probs_json | TEXT | JSON array 5-kelas fusion |
| dr_grade | INTEGER | Grade 0–4 |
| dr_label | TEXT | Label teks grade |
| confidence | REAL | Confidence % |
| doctor_status | TEXT | Pending/Approved/Rejected/Needs Review |
| doctor_notes | TEXT | Catatan dokter |
| clinical_rec_preset | TEXT | Rekomendasi klinis auto (dipisah ";") |
| clinical_rec_custom | TEXT | Rekomendasi klinis tambahan dokter |
| nonclinical_rec_preset | TEXT | Rekomendasi non-klinis auto |
| nonclinical_rec_custom | TEXT | Rekomendasi non-klinis tambahan |
| rec_timestamp | TEXT | Waktu simpan validasi |
| gradcam_filename | TEXT | **[XAI]** Nama file heatmap Grad-CAM |
| tabular_explanation_json | TEXT | **[XAI]** JSON feature risk contribution |

### Query Berguna

```sql
-- Lihat semua prediksi terbaru
SELECT id, patient_id, dr_grade, confidence, doctor_status, timestamp
FROM predictions ORDER BY timestamp DESC LIMIT 20;

-- Rekaman belum divalidasi
SELECT id, patient_id, dr_grade FROM predictions
WHERE doctor_status = 'Pending' OR doctor_status IS NULL;

-- Distribusi grade DR
SELECT dr_grade, dr_label, COUNT(*) as total
FROM predictions GROUP BY dr_grade ORDER BY dr_grade;

-- Rekaman dengan XAI tersedia
SELECT id, patient_id, gradcam_filename
FROM predictions WHERE gradcam_filename IS NOT NULL;
```

---

## 9. Melakukan Perubahan Lebih Lanjut

### 9.1 Mengganti Model

**Model CNN (EfficientNetB5):**
```
# Timpa file ini (pastikan arsitektur sama: 5 kelas output):
models/baseline_efficientnetb5_model_V2.pth
```

**Model KNN + Scaler:**
```
# Timpa kedua file ini bersama-sama:
models/model_knn.pkl
models/scaler_knn.pkl
# PENTING: scaler harus dilatih dengan 6 fitur yang sama
```

### 9.2 Mengubah Bobot Fusion

Edit `utils/fusion.py` — cari baris:
```python
W_IMG = 0.70   # bobot EfficientNetB5
W_TAB = 0.30   # bobot KNN
```

### 9.3 Mengubah Rekomendasi Otomatis

Edit `app.py` — dict `CLINICAL_RECS` dan `NONCLINICAL_RECS`:
```python
CLINICAL_RECS = {
    0: ["Rekomendasi Grade 0 baris 1", "..."],
    1: ["Rekomendasi Grade 1 baris 1", "..."],
    # dst...
}
```

### 9.4 Mengubah Ambang XAI Tabular

Edit `utils/tabular_handler.py` — dict `_RANGES`:
```python
_RANGES = {
    'hba1c': {
        'normal':  (0,   5.7,  'Normal',      '< 5.7%'),
        'warning': (5.7, 6.5,  'Pre-Diabetes', '5.7-6.4%'),
        'danger':  (6.5, 99,   'Diabetes',    '>= 6.5%'),
    },
    # dst...
}
```

### 9.5 Mengubah Tema UI

Edit `static/css/style.css` — CSS custom properties di `:root`:
```css
:root {
    --green-500: #22c55e;   /* warna utama */
    --green-700: #15803d;   /* warna teks */
    /* ubah sesuai kebutuhan */
}
```

### 9.6 Menambah Kolom Database Baru

Edit `database.py` — fungsi `init_db()`:
```python
# Tambahkan di blok migrasi:
migrations = [
    ('kolom_baru', 'TEXT'),   # tambahkan di sini
    ...
]
```

Lalu update fungsi `save_prediction()` dan `get_prediction()` sesuai kebutuhan.

### 9.7 Deployment Produksi (Windows)

```powershell
# Install Waitress (WSGI production server untuk Windows)
pip install waitress

# Jalankan dengan Waitress
python -c "from waitress import serve; from app import app; serve(app, host='0.0.0.0', port=5000)"
```

### 9.8 Deployment Produksi (Linux/Mac)

```bash
# Install Gunicorn
pip install gunicorn

# Jalankan dengan Gunicorn
gunicorn -w 1 -b 0.0.0.0:5000 app:app
```

> **Catatan:** Gunakan 1 worker (`-w 1`) karena model EfficientNetB5 di-load global. Multi-worker memerlukan refactor model loading ke dalam request context.

---

## 10. Troubleshooting

### Model gagal load saat startup
```
Pastikan file .pth dan .pkl ada di folder models/
Periksa kompatibilitas versi PyTorch dengan file .pth
```

### Error "Feature mismatch" saat KNN inference
```
scaler_knn.pkl dilatih dengan 6 fitur persis:
[age, hypertension, heart_disease, bmi, hba1c, blood_glucose]
Jangan ubah urutan di utils/tabular_handler.py
```

### Grad-CAM tidak muncul di halaman hasil
```
Cek log server — jika ada error Grad-CAM, halaman tetap tampil tanpa heatmap.
Penyebab umum: gambar fundus terlalu kecil atau rusak.
Cek folder uploads/ apakah file _gradcam.jpg terbuat.
```

### Database terkunci (locked)
```powershell
# Hentikan semua proses Python yang berjalan
Get-Process python | Stop-Process -Force
# Restart server
python app.py
```

### Port 5000 sudah dipakai
```powershell
# Cek PID yang menggunakan port 5000
netstat -ano | findstr ":5000"
# Matikan proses berdasarkan PID
Stop-Process -Id <PID> -Force
# Atau ganti port di app.py:
app.run(debug=True, port=5001, use_reloader=False)
```

### Reset database (hapus semua data)
```powershell
Remove-Item logs\validation_log.db
python app.py   # database akan dibuat ulang otomatis
```

---

## Menjalankan Pengujian

```powershell
# Test backend pipeline lengkap (termasuk inferensi)
python test_pipeline.py

# Test semua komponen XAI
python test_xai.py

# Test HTTP routes (server harus berjalan)
python test_http_routes.py
```

---

## Disclaimer

> ⚠️ **MIRA adalah alat bantu penelitian (CDSS prototype)**. Hasil prediksi AI bersifat **pendukung keputusan klinis**, bukan diagnosis definitif. Semua hasil harus divalidasi oleh tenaga medis yang berwenang sebelum digunakan untuk keputusan klinis.

---

*Dokumentasi terakhir diperbarui: Mei 2026 | Versi: 2.0 (dengan XAI)*
