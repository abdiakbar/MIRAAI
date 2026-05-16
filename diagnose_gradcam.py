"""diagnose_gradcam.py — Cari penyebab Grad-CAM gagal di server context"""
import sys, os, warnings, traceback
warnings.filterwarnings('ignore')
os.chdir('E:/Diabetic')
sys.path.insert(0, 'E:/Diabetic')

print("=== Grad-CAM Diagnosis ===\n")

from utils.image_handler import load_efficientnet_b5, generate_gradcam

model, device = load_efficientnet_b5()
img_path  = r'E:\Diabetic\aptos2019-blindness-detection\train_images\000c1434d8d7.png'
save_path = r'E:\Diabetic\uploads\diag_gradcam.jpg'
os.makedirs('uploads', exist_ok=True)

# Test 1: full_backward_hook
print("[Test 1] register_full_backward_hook tersedia?")
try:
    import torch
    dummy = torch.nn.Linear(2,2)
    h = dummy.register_full_backward_hook(lambda m,i,o: None)
    h.remove()
    print("    OK: full_backward_hook tersedia")
except Exception as e:
    print(f"    FAIL: {e}")

# Test 2: Grad-CAM dengan verbose error
print("\n[Test 2] generate_gradcam verbose...")
try:
    import torch, torch.nn.functional as F, cv2, numpy as np
    from utils.image_handler import circular_crop, apply_clahe, IMG_SIZE

    img = cv2.imread(img_path)
    img = circular_crop(img)
    img = apply_clahe(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_bgr = img.copy()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    arr = img_rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).to(device)
    print(f"    Tensor shape: {tensor.shape}, device: {tensor.device}")

    _acts = {}
    _grads = {}

    def _fwd(m, i, o): _acts['v'] = o
    def _bwd(m, gi, go): _grads['v'] = go[0]

    target_layer = model.features[-1]
    print(f"    Target layer: {type(target_layer).__name__}")

    h1 = target_layer.register_forward_hook(_fwd)
    h2 = target_layer.register_full_backward_hook(_bwd)

    model.zero_grad()
    with torch.enable_grad():
        out = model(tensor)
        print(f"    Output shape: {out.shape}")
        score = out[0, 2]
        print(f"    Score (class 2): {score.item():.4f}")
        score.backward()

    print(f"    Acts stored: {'v' in _acts}, Grads stored: {'v' in _grads}")
    if 'v' not in _grads:
        print("    FAIL: Gradients not captured — backward hook failed!")
    else:
        grads = _grads['v']
        acts  = _acts['v'].detach()
        print(f"    Grads shape: {grads.shape}, Acts shape: {acts.shape}")
        weights = grads.mean(dim=(2,3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam).squeeze().cpu().numpy()
        print(f"    CAM shape: {cam.shape}, min: {cam.min():.4f}, max: {cam.max():.4f}")
        mn, mx = cam.min(), cam.max()
        cam = (cam - mn) / (mx - mn + 1e-8)
        cam_r = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        hm = np.uint8(255 * cam_r)
        hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_bgr, 0.55, hm_color, 0.45, 0)
        cv2.imwrite(save_path, overlay)
        sz = os.path.getsize(save_path)
        print(f"    OK: heatmap saved ({sz} bytes) -> {save_path}")

    h1.remove(); h2.remove(); model.zero_grad()

except Exception as e:
    print(f"    ERROR: {e}")
    traceback.print_exc()

print("\n=== Diagnosis Selesai ===")
