"""
app.py — Flask backend DR Multi-Modal CDSS
Mengintegrasikan EfficientNetB5 + KNN + Late Fusion + XAI (Grad-CAM & Feature Risk).
"""
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

import os
import uuid
import json
import traceback

# ── File-based logger (bypass stdout encoding/buffering issues) ───────────────
_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', 'xai_debug.log')

def _xai_log(msg):
    """Tulis log ke file agar tidak tergantung pada stdout encoding."""
    try:
        os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)
        with open(_LOG_PATH, 'a', encoding='utf-8') as f:
            from datetime import datetime
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
    except Exception:
        pass

from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory

from utils.image_handler  import predict_fundus, load_efficientnet_b5, generate_gradcam
from utils.tabular_handler import process_tabular, explain_tabular
from utils.fusion          import semantic_fusion
from database import (
    init_db, save_prediction, save_validation,
    get_prediction, get_all_predictions, delete_prediction
)

# ─── App init ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = 'dr-cdss-secret-2024-xk9q'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER']       = os.path.join(BASE_DIR, 'uploads')
app.config['MAX_CONTENT_LENGTH']  = 16 * 1024 * 1024

ALLOWED_EXT = {'png', 'jpg', 'jpeg', 'tiff', 'bmp'}

# ─── Rekomendasi Otomatis per DR Grade ────────────────────────────────────────
CLINICAL_RECS = {
    0: ["Pemeriksaan mata rutin setiap 12 bulan",
        "Pertahankan kontrol gula darah optimal (HbA1c < 7%)",
        "Monitoring tekanan darah rutin"],
    1: ["Follow-up ophthalmologi dalam 6 bulan",
        "Optimasi kontrol gula darah ketat",
        "Konsultasi dokter spesialis mata",
        "Monitoring tekanan darah secara berkala"],
    2: ["Rujukan ke dokter ophthalmolog segera",
        "Follow-up dalam 3–4 bulan",
        "Pertimbangkan laser photocoagulation fokal",
        "Optimasi terapi diabetes intensif"],
    3: ["Rujukan SEGERA ke ophthalmolog",
        "Pertimbangkan laser scatter (PRP)",
        "Kontrol gula darah sangat ketat",
        "Monitoring fungsi ginjal",
        "Evaluasi tekanan intraokular"],
    4: ["Rujukan SEGERA ke spesialis Vitreoretinal",
        "Evaluasi untuk injeksi anti-VEGF (Bevacizumab/Ranibizumab)",
        "Pertimbangkan vitrektomi (jika ada perdarahan vitreous)",
        "Panretinal photocoagulation (PRP) laser",
        "Kontrol gula darah intensif segera"],
}

NONCLINICAL_RECS = {
    0: ["Edukasi diet rendah gula dan karbohidrat sederhana",
        "Olahraga aerobik minimal 150 menit per minggu",
        "Monitoring gula darah mandiri (self-monitoring)",
        "Jadwal tidur teratur (7–8 jam/malam)"],
    1: ["Edukasi diet rendah gula dan karbohidrat sederhana",
        "Olahraga aerobik minimal 150 menit per minggu",
        "Monitoring gula darah mandiri (self-monitoring)",
        "Hindari merokok dan konsumsi alkohol berlebihan",
        "Penggunaan kacamata pelindung UV saat outdoor"],
    2: ["Edukasi diet rendah gula dan karbohidrat sederhana",
        "Olahraga aerobik minimal 150 menit per minggu",
        "Monitoring gula darah mandiri (self-monitoring)",
        "Hindari merokok dan konsumsi alkohol berlebihan",
        "Penggunaan kacamata pelindung UV saat outdoor",
        "Manajemen stres: meditasi, yoga, atau relaksasi"],
    3: ["Edukasi diet rendah gula dan karbohidrat sederhana",
        "Batasi aktivitas fisik berat sementara",
        "Monitoring gula darah mandiri lebih ketat",
        "Hindari merokok dan konsumsi alkohol berlebihan",
        "Penggunaan kacamata pelindung UV saat outdoor",
        "Manajemen stres: meditasi, yoga, atau relaksasi",
        "Konsumsi suplemen antioksidan (Vit C & E) sesuai anjuran dokter"],
    4: ["Istirahat total aktivitas fisik berat",
        "Diet ketat rendah gula, konsultasi ahli gizi",
        "Monitoring gula darah mandiri intensif",
        "Hindari merokok dan konsumsi alkohol",
        "Jaga kebersihan mata, hindari paparan debu",
        "Manajemen stres: meditasi, yoga, atau relaksasi",
        "Konsumsi suplemen antioksidan (Vit C & E) sesuai anjuran dokter"],
}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # ── Validasi file ──
    if 'fundus_image' not in request.files:
        flash('File gambar fundus tidak ditemukan.', 'error')
        return redirect(url_for('index'))

    file = request.files['fundus_image']
    if file.filename == '' or not allowed_file(file.filename):
        flash('Harap upload file gambar yang valid (PNG, JPG, JPEG, TIFF, BMP).', 'error')
        return redirect(url_for('index'))

    # ── Validasi form ──
    try:
        patient_id      = request.form.get('patient_id', '').strip()
        if not patient_id:
            flash('ID Pasien wajib diisi.', 'error')
            return redirect(url_for('index'))

        patient_name    = request.form.get('patient_name', '').strip()
        gender          = int(request.form['gender'])
        age             = float(request.form['age'])
        hypertension    = int(request.form['hypertension'])
        heart_disease   = int(request.form['heart_disease'])
        smoking_history = int(request.form['smoking_history'])
        hba1c           = float(request.form['hba1c'])
        blood_glucose   = float(request.form['blood_glucose'])
        tinggi_badan    = float(request.form['tinggi_badan'])
        berat_badan     = float(request.form['berat_badan'])
        if tinggi_badan <= 0:
            raise ValueError("Tinggi badan tidak valid")
        bmi = round(berat_badan / ((tinggi_badan / 100) ** 2), 2)

    except (KeyError, ValueError) as e:
        flash(f'Data form tidak lengkap atau tidak valid: {e}', 'error')
        return redirect(url_for('index'))

    # ── Simpan gambar ──
    ext = file.filename.rsplit('.', 1)[1].lower()
    img_uuid     = uuid.uuid4().hex
    img_filename = f"{img_uuid}.{ext}"
    img_path     = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(img_path)

    # ── Inferensi ──
    try:
        prob_img = predict_fundus(img_path, model_img, device)
        prob_tab = process_tabular(
            gender, age, hypertension, heart_disease,
            smoking_history, bmi, hba1c, blood_glucose
        )
        fusion = semantic_fusion(prob_img, prob_tab)
    except Exception as e:
        flash(f'Error saat inferensi model: {e}', 'error')
        return redirect(url_for('index'))

    grade = fusion['dr_grade']

    # ── XAI 1: Grad-CAM heatmap ──────────────────────────────────────────────────────
    gradcam_filename = None
    try:
        gc_filename = f"{img_uuid}_gradcam.jpg"
        gc_path     = os.path.join(app.config['UPLOAD_FOLDER'], gc_filename)
        _xai_log(f'Generating Grad-CAM: class={grade}, path={gc_path}')
        result = generate_gradcam(img_path, model_img, device, grade, gc_path)
        if result:
            gradcam_filename = gc_filename
            _xai_log(f'Grad-CAM OK: {gc_filename}')
        else:
            _xai_log('Grad-CAM returned None (internal error)')
    except Exception as e:
        _xai_log(f'Grad-CAM EXCEPTION: {traceback.format_exc()}')

    # ── XAI 2: Feature risk contribution ───────────────────────────────────────────────
    tab_explanation = None
    try:
        _xai_log(f'explain_tabular: age={age}, htn={hypertension}, bmi={bmi}, hba1c={hba1c}, bg={blood_glucose}')
        tab_explanation = explain_tabular(
            age, hypertension, heart_disease, bmi, hba1c, blood_glucose
        )
        _xai_log(f'explain_tabular OK: {len(tab_explanation)} features')
    except Exception as e:
        _xai_log(f'explain_tabular EXCEPTION: {traceback.format_exc()}')

    # ── Simpan ke DB ──
    pred_id = save_prediction(
        patient_id=patient_id, patient_name=patient_name,
        img_filename=img_filename,
        gender=gender, age=age,
        hypertension=hypertension, heart_disease=heart_disease,
        smoking_history=smoking_history, bmi=bmi,
        hba1c=hba1c, blood_glucose=blood_glucose,
        knn_probs=prob_tab,
        img_probs=prob_img,
        fusion_probs=fusion['combined_probs'],
        dr_grade=grade,
        dr_label=fusion['dr_label'],
        confidence=fusion['confidence'],
        tinggi_badan=tinggi_badan,
        berat_badan=berat_badan,
        gradcam_filename=gradcam_filename,
        tabular_explanation=tab_explanation,
    )

    return redirect(url_for('result', pred_id=pred_id))


@app.route('/result/<int:pred_id>')
def result(pred_id):
    data = get_prediction(pred_id)
    if not data:
        flash('Rekaman prediksi tidak ditemukan.', 'error')
        return redirect(url_for('index'))
    data['img_probs']    = json.loads(data['img_probs_json'])
    data['fusion_probs'] = json.loads(data['fusion_probs_json'])
    # Parse tabular XAI JSON jika ada
    tab_xai = None
    if data.get('tabular_explanation_json'):
        try:
            tab_xai = json.loads(data['tabular_explanation_json'])
        except Exception:
            tab_xai = None
    return render_template('result.html', data=data, tab_xai=tab_xai)


@app.route('/validate/<int:pred_id>', methods=['GET', 'POST'])
def validate(pred_id):
    data = get_prediction(pred_id)
    if not data:
        flash('Rekaman tidak ditemukan.', 'error')
        return redirect(url_for('history'))

    if request.method == 'POST':
        doctor_status      = request.form.get('doctor_status', 'Pending')
        doctor_notes       = request.form.get('doctor_notes', '').strip()
        clinical_custom    = request.form.get('clinical_custom', '').strip()
        nonclinical_custom = request.form.get('nonclinical_custom', '').strip()

        grade              = data['dr_grade']
        clinical_preset    = '; '.join(CLINICAL_RECS.get(grade, []))
        nonclinical_preset = '; '.join(NONCLINICAL_RECS.get(grade, []))

        save_validation(
            pred_id=pred_id,
            doctor_status=doctor_status,
            doctor_notes=doctor_notes,
            clinical_preset=clinical_preset,
            clinical_custom=clinical_custom,
            nonclinical_preset=nonclinical_preset,
            nonclinical_custom=nonclinical_custom,
        )
        flash('Validasi berhasil disimpan.', 'success')
        return redirect(url_for('history'))

    data['img_probs']    = json.loads(data['img_probs_json'])
    data['fusion_probs'] = json.loads(data['fusion_probs_json'])
    grade            = data['dr_grade']
    auto_clinical    = CLINICAL_RECS.get(grade, [])
    auto_nonclinical = NONCLINICAL_RECS.get(grade, [])

    return render_template('validate.html', data=data,
                           auto_clinical=auto_clinical,
                           auto_nonclinical=auto_nonclinical)


@app.route('/xai_debug')
def xai_debug():
    """Route diagnostik — test XAI functions langsung di Flask context."""
    from flask import jsonify
    import traceback as tb
    result = {'gradcam': None, 'tabular': None, 'errors': []}

    # Test explain_tabular
    try:
        expl = explain_tabular(55.0, 1, 0, 28.5, 7.8, 195.0)
        result['tabular'] = {'status': 'OK', 'features': len(expl), 'top': expl[0]['name']}
    except Exception as e:
        result['errors'].append({'tabular': tb.format_exc()})

    # Test Grad-CAM
    try:
        import glob
        imgs = glob.glob(os.path.join(BASE_DIR, 'uploads', '*.png')) + \
               glob.glob(os.path.join(BASE_DIR, 'uploads', '*.jpg'))
        # Filter out gradcam files
        imgs = [i for i in imgs if '_gradcam' not in i]
        if imgs:
            test_img = imgs[0]
            test_out = test_img.replace('.png', '_dbg.jpg').replace('.jpg', '_dbg.jpg')
            gc = generate_gradcam(test_img, model_img, device, 2, test_out)
            result['gradcam'] = {'status': 'OK' if gc else 'returned_None', 'img': os.path.basename(test_img)}
            if os.path.exists(test_out): os.remove(test_out)
        else:
            result['gradcam'] = {'status': 'no_test_image'}
    except Exception as e:
        result['errors'].append({'gradcam': tb.format_exc()})

    # Read XAI log file
    try:
        if os.path.exists(_LOG_PATH):
            with open(_LOG_PATH, encoding='utf-8') as f:
                result['xai_log'] = f.read()[-3000:]
    except Exception:
        pass

    return jsonify(result)


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/history')
def history():
    records = get_all_predictions()
    return render_template('history.html', records=records)


@app.route('/record/<int:pred_id>')
def record(pred_id):
    data = get_prediction(pred_id)
    if not data:
        flash('Rekaman tidak ditemukan.', 'error')
        return redirect(url_for('history'))
    data['img_probs']    = json.loads(data['img_probs_json'])
    data['fusion_probs'] = json.loads(data['fusion_probs_json'])
    tab_xai = None
    if data.get('tabular_explanation_json'):
        try:
            tab_xai = json.loads(data['tabular_explanation_json'])
        except Exception:
            tab_xai = None
    return render_template('result.html', data=data, tab_xai=tab_xai, is_record=True)


@app.route('/delete/<int:pred_id>', methods=['POST'])
def delete_record(pred_id):
    """Hard delete — hapus record + file gambar + file Grad-CAM."""
    img_file, gradcam_file = delete_prediction(pred_id)
    for fname in [img_file, gradcam_file]:
        if fname:
            fpath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
            if os.path.exists(fpath):
                os.remove(fpath)
    flash(f'Rekaman #{pred_id} berhasil dihapus.', 'success')
    return redirect(url_for('history'))


# ─── Startup ──────────────────────────────────────────────────────────────────
os.makedirs(os.path.join(BASE_DIR, 'uploads'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'logs'), exist_ok=True)
init_db()

print('[DR-CDSS] Loading EfficientNetB5 model...')
model_img, device = load_efficientnet_b5()
print(f'[DR-CDSS] Model ready on: {device}')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)