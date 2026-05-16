/* main.js — DR CDSS interactions & validation */

document.addEventListener('DOMContentLoaded', () => {
  // ── 1. Auto-dismiss flash messages ─────────────────────────────
  document.querySelectorAll('.flash').forEach(el => {
    setTimeout(() => {
      el.style.transition = 'opacity .4s';
      el.style.opacity = '0';
      setTimeout(() => el.remove(), 400);
    }, 5000);
  });

  // ── 2. Form Validation (Heuristic #5: Error Prevention) ────────
  const form = document.getElementById('predict-form');
  if (form) {
    form.addEventListener('submit', (e) => {
      let firstError = null;
      
      // Clear previous errors
      form.querySelectorAll('.form-group').forEach(group => {
        group.classList.remove('has-error');
        const errText = group.querySelector('.form-error');
        if (errText) errText.remove();
      });

      // Validate required fields
      form.querySelectorAll('[required]').forEach(input => {
        if (!input.value.trim() || (input.type === 'file' && !input.files.length)) {
          e.preventDefault();
          const group = input.closest('.form-group') || input.closest('.card') || input.parentElement;
          group.classList.add('has-error');
          input.setAttribute('aria-invalid', 'true');
          
          if (!group.querySelector('.form-error')) {
            const err = document.createElement('span');
            err.className = 'form-error';
            err.innerHTML = '<span aria-hidden="true">⚠️</span> Bidang ini wajib diisi';
            group.appendChild(err);
          }
          
          if (!firstError) firstError = input;
        } else {
          input.removeAttribute('aria-invalid');
        }
      });

      if (firstError) {
        firstError.focus();
      }
    });

    // Clear error on input
    form.querySelectorAll('input, select, textarea').forEach(input => {
      input.addEventListener('input', () => {
        const group = input.closest('.form-group') || input.closest('.card') || input.parentElement;
        group.classList.remove('has-error');
        input.removeAttribute('aria-invalid');
        const errText = group.querySelector('.form-error');
        if (errText) errText.remove();
      });
    });
  }

  // ── 3. Upload Zone Enhancement ──────────────────────────────────
  const zone = document.getElementById('upload-zone');
  const input = document.getElementById('fundus-input');
  const preview = document.getElementById('preview-img');
  const placeholder = document.getElementById('upload-placeholder');
  const previewWrap = document.getElementById('upload-preview');

  if (zone && input) {
    zone.addEventListener('click', () => input.click());
    zone.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        input.click();
      }
    });

    input.addEventListener('change', () => {
      if (input.files && input.files[0]) {
        const reader = new FileReader();
        reader.onload = (e) => {
          preview.src = e.target.result;
          placeholder.style.display = 'none';
          previewWrap.style.display = 'block';
        };
        reader.readAsDataURL(input.files[0]);
      }
    });

    // Drag and drop
    zone.addEventListener('dragover', (e) => {
      e.preventDefault();
      zone.classList.add('dragover');
    });
    zone.addEventListener('dragleave', () => {
      zone.classList.remove('dragover');
    });
    zone.addEventListener('drop', (e) => {
      e.preventDefault();
      zone.classList.remove('dragover');
      if (e.dataTransfer.files.length) {
        input.files = e.dataTransfer.files;
        input.dispatchEvent(new Event('change'));
      }
    });
  }
});
