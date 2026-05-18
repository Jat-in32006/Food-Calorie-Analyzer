/* ============================================================
   NutriScan AI — Frontend Logic
   ============================================================ */

const PALETTE = [
    '#10b981','#3b82f6','#f59e0b','#f43f5e','#8b5cf6',
    '#06b6d4','#ec4899','#84cc16','#fb923c','#a78bfa'
];

// ── DOM refs ──────────────────────────────────────────────────
const dropZone       = document.getElementById('drop-zone');
const fileInput      = document.getElementById('file-input');
const uploadPrompt   = document.getElementById('upload-prompt');
const uploadPreview  = document.getElementById('upload-preview');
const previewImg     = document.getElementById('preview-img');
const analyzeBtn     = document.getElementById('analyze-btn');
const btnLabel       = document.getElementById('btn-label');
const statusDot      = document.getElementById('status-dot');
const statusText     = document.getElementById('status-text');
const loadingOverlay = document.getElementById('loading-overlay');
const loadingStepTxt = document.getElementById('loading-step-text');
const canvasPlaceholder = document.getElementById('canvas-placeholder');
const canvas         = document.getElementById('detection-canvas');
const ctx            = canvas.getContext('2d');
const tableBody      = document.getElementById('table-body');
const globalCount    = document.getElementById('global-count');
const globalCalories = document.getElementById('global-calories');
const itemCountBadge = document.getElementById('item-count-badge');
const breakdownSection = document.getElementById('calorie-breakdown');
const breakdownBars  = document.getElementById('breakdown-bars');

const PIPELINE_STEPS = [
    { id: 'step-1', label: 'SVD Preprocessing...' },
    { id: 'step-2', label: 'CLAHE Enhancement...' },
    { id: 'step-3', label: 'Watershed Segmentation...' },
    { id: 'step-4', label: 'YOLOv5 Detection...' },
    { id: 'step-5', label: 'Calorie Mapping...' },
];

let currentFile = null;
let stepTimer   = null;

// ── Upload / Drag-Drop ────────────────────────────────────────
dropZone.addEventListener('click', () => fileInput.click());

['dragenter','dragover','dragleave','drop'].forEach(ev =>
    dropZone.addEventListener(ev, e => { e.preventDefault(); e.stopPropagation(); })
);

['dragenter','dragover'].forEach(ev =>
    dropZone.addEventListener(ev, () => dropZone.classList.add('dragover'))
);

['dragleave','drop'].forEach(ev =>
    dropZone.addEventListener(ev, () => dropZone.classList.remove('dragover'))
);

dropZone.addEventListener('drop', e => {
    const files = e.dataTransfer.files;
    if (files.length) handleFile(files[0]);
});

fileInput.addEventListener('change', e => {
    if (e.target.files.length) handleFile(e.target.files[0]);
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        showToast('Please upload an image file (JPG, PNG, WEBP).', 'error');
        return;
    }
    currentFile = file;

    const reader = new FileReader();
    reader.onload = e => {
        previewImg.src = e.target.result;
        uploadPrompt.style.display = 'none';
        uploadPreview.style.display = 'block';
        analyzeBtn.disabled = false;

        // Show image on canvas immediately
        showImageOnCanvas(e.target.result);
    };
    reader.readAsDataURL(file);
}

// ── Canvas Preview ────────────────────────────────────────────
function showImageOnCanvas(dataUrl) {
    canvasPlaceholder.style.display = 'none';
    const img = new Image();
    img.onload = () => {
        canvas.width  = img.naturalWidth;
        canvas.height = img.naturalHeight;
        ctx.drawImage(img, 0, 0);
    };
    img.src = dataUrl;
}

// ── Analyze Button ────────────────────────────────────────────
analyzeBtn.addEventListener('click', () => {
    if (!currentFile) return;
    runAnalysis(currentFile);
});

async function runAnalysis(file) {
    // UI: loading state
    analyzeBtn.disabled = true;
    btnLabel.textContent = 'Analyzing...';
    loadingOverlay.style.display = 'flex';
    setStatus('active', 'Analyzing...');
    resetPipelineSteps();
    startPipelineAnimation();

    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch('/api/analyze', { method: 'POST', body: formData });
        const data = await res.json();

        stopPipelineAnimation();
        markAllStepsDone();

        if (data.status === 'success') {
            renderResults(data, previewImg.src);
            setStatus('active', 'Analysis Complete');
        } else {
            showToast(data.error || 'Analysis failed.', 'error');
            setStatus('error', 'Failed');
        }
    } catch (err) {
        console.error(err);
        stopPipelineAnimation();
        showToast('Server error — make sure the backend is running.', 'error');
        setStatus('error', 'Error');
    } finally {
        loadingOverlay.style.display = 'none';
        analyzeBtn.disabled = false;
        btnLabel.textContent = 'Analyze Calories';
    }
}

// ── Render Results ────────────────────────────────────────────
function renderResults(data, imageDataUrl) {
    const objects = data.detected_objects || [];
    const total   = data.estimated_calories_total || 0;
    const count   = data.overall_count || 0;

    // Draw bounding boxes on canvas
    drawDetections(imageDataUrl, objects);

    // Animate metrics
    animateNumber(globalCount,    0, count, 900);
    animateNumber(globalCalories, 0, total, 1100, ' kcal');

    // Table
    renderTable(objects);

    // Calorie breakdown
    renderBreakdown(objects, total);
}

function drawDetections(imageDataUrl, objects) {
    const img = new Image();
    img.onload = () => {
        canvas.width  = img.naturalWidth;
        canvas.height = img.naturalHeight;
        ctx.drawImage(img, 0, 0);

        const scale = Math.max(1, Math.floor(canvas.width / 400));

        objects.forEach((item, i) => {
            const [x, y, w, h] = item.bbox;
            const color = PALETTE[i % PALETTE.length];

            // Box shadow glow
            ctx.shadowColor = color;
            ctx.shadowBlur  = 12;

            // Bounding box
            ctx.strokeStyle = color;
            ctx.lineWidth   = scale * 2;
            ctx.strokeRect(x, y, w, h);

            ctx.shadowBlur = 0;

            // Label background
            const fontSize  = Math.max(12, scale * 13);
            ctx.font        = `bold ${fontSize}px Inter, sans-serif`;
            const label     = `${item.class} · ${item.calories > 0 ? item.calories + ' kcal' : '?'}`;
            const tw        = ctx.measureText(label).width;
            const pad       = fontSize * 0.45;
            const bh        = fontSize + pad * 2;

            // Pill background
            ctx.fillStyle = color;
            roundRect(ctx, x, y - bh - 2, tw + pad * 2, bh, 4);
            ctx.fill();

            // Label text
            ctx.fillStyle = '#ffffff';
            ctx.fillText(label, x + pad, y - pad - 2);

            // Instance ID circle
            const circR = scale * 9;
            ctx.beginPath();
            ctx.arc(x + w - circR - 4, y + circR + 4, circR, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();
            ctx.fillStyle = '#fff';
            ctx.font = `bold ${circR}px Inter, sans-serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(item.id, x + w - circR - 4, y + circR + 4);
            ctx.textAlign = 'left';
            ctx.textBaseline = 'alphabetic';
        });
    };
    img.src = imageDataUrl;
}

function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
}

// ── Table ─────────────────────────────────────────────────────
function renderTable(objects) {
    tableBody.innerHTML = '';

    if (!objects.length) {
        tableBody.innerHTML = `
            <tr><td colspan="5">
                <div class="empty-state"><span>🔍</span><p>No food items detected</p></div>
            </td></tr>`;
        itemCountBadge.style.display = 'none';
        return;
    }

    itemCountBadge.textContent = `${objects.length} item${objects.length !== 1 ? 's' : ''}`;
    itemCountBadge.style.display = 'inline-block';

    objects.forEach((item, i) => {
        const color = PALETTE[i % PALETTE.length];
        const conf  = (item.confidence * 100).toFixed(1);
        const isZero = item.calories === 0;

        const tr = document.createElement('tr');
        tr.className = 'row-enter';
        tr.style.animationDelay = `${i * 60}ms`;

        tr.innerHTML = `
            <td><strong style="color:${color}">#${item.id}</strong></td>
            <td>
                <span class="food-chip">
                    <span class="food-chip-dot" style="background:${color}"></span>
                    ${escHtml(item.class)}
                </span>
            </td>
            <td>
                <div class="conf-bar-wrap">
                    <div class="conf-bar">
                        <div class="conf-bar-fill" style="width:${conf}%"></div>
                    </div>
                    <span class="conf-text">${conf}%</span>
                </div>
            </td>
            <td>
                <span class="cal-badge ${isZero ? 'zero' : ''}">
                    ${isZero ? '—' : item.calories + ' kcal'}
                </span>
            </td>
            <td><span class="area-text">${Math.round(item.area).toLocaleString()}</span></td>
        `;
        tableBody.appendChild(tr);
    });
}

// ── Calorie Breakdown ─────────────────────────────────────────
function renderBreakdown(objects, total) {
    const withCals = objects.filter(o => o.calories > 0);
    if (!withCals.length || total === 0) {
        breakdownSection.style.display = 'none';
        return;
    }

    breakdownBars.innerHTML = '';
    breakdownSection.style.display = 'block';

    withCals.forEach((item, i) => {
        const pct   = Math.round((item.calories / total) * 100);
        const color = PALETTE[i % PALETTE.length];

        const row = document.createElement('div');
        row.className = 'breakdown-row';
        row.innerHTML = `
            <span class="breakdown-label" title="${escHtml(item.class)}">${escHtml(item.class)}</span>
            <div class="breakdown-bar-wrap">
                <div class="breakdown-bar-fill" style="width:0%; background:linear-gradient(90deg,${color},${color}aa)"></div>
            </div>
            <span class="breakdown-kcal">${item.calories} kcal</span>
        `;
        breakdownBars.appendChild(row);

        // Animate bar width
        requestAnimationFrame(() => {
            setTimeout(() => {
                row.querySelector('.breakdown-bar-fill').style.width = pct + '%';
            }, i * 80 + 100);
        });
    });
}

// ── Pipeline Step Animation ───────────────────────────────────
let currentStep = 0;

function resetPipelineSteps() {
    currentStep = 0;
    PIPELINE_STEPS.forEach(s => {
        const el = document.getElementById(s.id);
        el.classList.remove('active', 'done');
    });
}

function startPipelineAnimation() {
    currentStep = 0;
    activateStep(0);
}

function activateStep(idx) {
    if (idx >= PIPELINE_STEPS.length) return;

    // Mark previous as done
    if (idx > 0) {
        document.getElementById(PIPELINE_STEPS[idx - 1].id).classList.remove('active');
        document.getElementById(PIPELINE_STEPS[idx - 1].id).classList.add('done');
    }

    const step = PIPELINE_STEPS[idx];
    document.getElementById(step.id).classList.add('active');
    loadingStepTxt.textContent = step.label;

    stepTimer = setTimeout(() => activateStep(idx + 1), 900);
}

function stopPipelineAnimation() {
    clearTimeout(stepTimer);
}

function markAllStepsDone() {
    PIPELINE_STEPS.forEach(s => {
        const el = document.getElementById(s.id);
        el.classList.remove('active');
        el.classList.add('done');
    });
}

// ── Status Pill ───────────────────────────────────────────────
function setStatus(state, text) {
    statusDot.className = 'status-dot';
    if (state === 'active') statusDot.classList.add('active');
    if (state === 'error')  statusDot.classList.add('error');
    statusText.textContent = text;
}

// ── Number Animation ──────────────────────────────────────────
function animateNumber(el, from, to, duration, suffix = '') {
    const start = performance.now();
    function step(now) {
        const p = Math.min((now - start) / duration, 1);
        const ease = 1 - Math.pow(1 - p, 3); // ease-out cubic
        el.textContent = Math.round(from + (to - from) * ease) + suffix;
        if (p < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
}

// ── Toast Notification ────────────────────────────────────────
function showToast(msg, type = 'info') {
    const existing = document.getElementById('toast');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.id = 'toast';
    toast.textContent = msg;
    Object.assign(toast.style, {
        position: 'fixed', bottom: '24px', right: '24px',
        background: type === 'error' ? 'rgba(244,63,94,0.9)' : 'rgba(16,185,129,0.9)',
        color: '#fff', padding: '12px 20px', borderRadius: '10px',
        fontSize: '0.88rem', fontWeight: '600', zIndex: '9999',
        boxShadow: '0 4px 20px rgba(0,0,0,0.4)',
        animation: 'fadeSlideIn 0.3s ease',
        maxWidth: '320px', lineHeight: '1.4',
    });
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
}

// ── Utility ───────────────────────────────────────────────────
function escHtml(str) {
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}
