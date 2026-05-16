"""
utils/image_handler.py — Pipeline citra fundus + Grad-CAM XAI
Preprocessing: Circular Crop → CLAHE (LAB) → 128×128 → normalisasi /255
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

IMG_SIZE = 128


# ─── Preprocessing ────────────────────────────────────────────────────────────

def circular_crop(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return img
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    return img[y:y+h, x:x+w]


def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _preprocess(img_path):
    """Shared preprocessing — return (img_rgb_np, img_bgr_np, tensor)."""
    img = cv2.imread(img_path)
    img = circular_crop(img)
    img = apply_clahe(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_bgr = img.copy()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    arr = img_rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return img_rgb, img_bgr, tensor


# ─── Model loader ─────────────────────────────────────────────────────────────

def load_efficientnet_b5():
    model = models.efficientnet_b5(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(
        torch.load('models/baseline_efficientnetb5_model_V2.pth', map_location=device, weights_only=False)
    )
    model.eval()
    return model, device


# ─── Inference ────────────────────────────────────────────────────────────────

def predict_fundus(path, model, device):
    """Return softmax probabilities (5 kelas DR)."""
    _, _, tensor = _preprocess(path)
    tensor = tensor.to(device)
    with torch.no_grad():
        output = model(tensor)
        return torch.softmax(output, dim=1).cpu().numpy()[0]


# ─── Grad-CAM XAI ─────────────────────────────────────────────────────────────

def generate_gradcam(img_path, model, device, target_class, save_path):
    """
    Buat heatmap Grad-CAM untuk kelas prediksi dan simpan sebagai overlay
    di atas citra fundus yang sudah dipreprocess.

    Args:
        img_path     : path ke file gambar fundus asli
        model        : EfficientNetB5 yang sudah di-load
        device       : torch.device
        target_class : int (0–4) — kelas DR yang diprediksi
        save_path    : path untuk menyimpan gambar overlay hasil Grad-CAM

    Returns:
        save_path jika berhasil, None jika gagal
    """
    img_rgb, img_bgr, tensor = _preprocess(img_path)
    tensor = tensor.to(device)

    # Storage untuk hook
    _acts  = {}
    _grads = {}

    def _fwd(module, inp, out):
        _acts['v'] = out

    def _bwd(module, g_in, g_out):
        _grads['v'] = g_out[0]

    # Hook pada layer features terakhir EfficientNetB5
    target_layer = model.features[-1]
    h1 = target_layer.register_forward_hook(_fwd)
    h2 = target_layer.register_full_backward_hook(_bwd)

    try:
        model.zero_grad()

        # Forward pass dengan gradient aktif
        with torch.enable_grad():
            out   = model(tensor)
            score = out[0, int(target_class)]
            score.backward()

        # Grad-CAM: rata-rata gradien per channel → bobot
        grads   = _grads['v']                              # (1, C, H, W)
        acts    = _acts['v'].detach()                      # (1, C, H, W)
        weights = grads.mean(dim=(2, 3), keepdim=True)    # (1, C, 1, 1)

        cam = (weights * acts).sum(dim=1, keepdim=True)   # (1, 1, H, W)
        cam = F.relu(cam).squeeze().cpu().numpy()          # (H, W)

        # Normalisasi 0–1
        mn, mx = cam.min(), cam.max()
        if mx - mn > 1e-8:
            cam = (cam - mn) / (mx - mn)
        else:
            cam = np.zeros_like(cam)

        # Resize ke IMG_SIZE dan terapkan colormap
        cam_resized   = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        heatmap       = np.uint8(255 * cam_resized)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Overlay pada citra hasil preprocess (bukan asli)
        overlay = cv2.addWeighted(img_bgr, 0.55, heatmap_color, 0.45, 0)
        cv2.imwrite(save_path, overlay)
        return save_path

    except Exception as exc:
        print(f'[Grad-CAM] Gagal: {exc}')
        return None

    finally:
        h1.remove()
        h2.remove()
        model.zero_grad()