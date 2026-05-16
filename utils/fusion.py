"""
utils/fusion.py — Late Fusion: Semantic Projection + Weighted Average
Menggabungkan output EfficientNetB5 (5 kelas) dengan KNN (2 kelas).
"""

import numpy as np

DR_LABELS = [
    'No DR',
    'Mild DR',
    'Moderate DR',
    'Severe DR',
    'Proliferative DR'
]

# Distribusi bobot proyeksi: jika pasien diabetes,
# lebih mungkin mengalami DR ringan–sedang daripada langsung parah.
DIABETES_GRADE_DIST = [0.0, 0.40, 0.30, 0.20, 0.10]


def semantic_fusion(prob_img, prob_tab, w_img=0.70, w_tab=0.30):
    """
    Fusi semantik probabilitas dari dua model.

    Args:
        prob_img : array-like (5,)  — probabilitas DR grade 0-4 dari EfficientNetB5
        prob_tab : array-like (2,)  — [P(no_diabetes), P(diabetes)] dari KNN
        w_img    : float — bobot model image (default 0.70)
        w_tab    : float — bobot model tabular (default 0.30)

    Returns:
        dict:
            combined_probs  : list[float] (5) — probabilitas fusi yang dinormalisasi
            risk_knn        : list[float] (5) — proyeksi KNN ke 5 dimensi
            dr_grade        : int             — grade DR final (argmax)
            dr_label        : str             — label DR final
            confidence      : float           — confidence dalam persen
    """
    p_nd = float(prob_tab[0])   # P(no diabetes)
    p_d  = float(prob_tab[1])   # P(diabetes)

    # --- Proyeksi semantik KNN → 5 dimensi ---
    # Grade 0 (No DR) dikaitkan dengan P(no_diabetes)
    # Grade 1-4 dikaitkan dengan P(diabetes) * distribusi keparahan
    risk_knn = np.array([
        p_nd,
        p_d * DIABETES_GRADE_DIST[1],
        p_d * DIABETES_GRADE_DIST[2],
        p_d * DIABETES_GRADE_DIST[3],
        p_d * DIABETES_GRADE_DIST[4],
    ], dtype=np.float32)

    # Normalisasi agar risk_knn membentuk distribusi probabilitas valid
    risk_knn = risk_knn / risk_knn.sum()

    # --- Weighted average fusion ---
    img_arr = np.array(prob_img, dtype=np.float32)
    combined = w_img * img_arr + w_tab * risk_knn

    # Normalisasi hasil akhir (seharusnya sudah ~1.0, tapi untuk numerik safety)
    combined = combined / combined.sum()

    dr_grade = int(np.argmax(combined))
    confidence = float(combined[dr_grade]) * 100.0
    dr_label = DR_LABELS[dr_grade]

    return {
        'combined_probs': combined.tolist(),
        'risk_knn':       risk_knn.tolist(),
        'dr_grade':       dr_grade,
        'dr_label':       dr_label,
        'confidence':     confidence,
    }
