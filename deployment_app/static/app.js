/* =========================================
   NEWSBIAS AI — Enhanced Frontend Logic
   app.js
   ========================================= */

// ----------------------------------------
// 1. CHARACTER COUNTER
// ----------------------------------------
const textInput = document.getElementById('text-input');
const charCount = document.getElementById('char-count');

if (textInput && charCount) {
    textInput.addEventListener('input', () => {
        const len = textInput.value.length;
        charCount.textContent = `${len} character${len !== 1 ? 's' : ''}`;
    });
}

// ----------------------------------------
// 2. SAMPLE TEXTS (Hindi news examples)
// ----------------------------------------
const SAMPLES = {
    1: 'प्रधानमंत्री नरेंद्र मोदी ने आज देश के नागरिकों से स्वच्छता अभियान में भाग लेने की अपील की और कहा कि स्वच्छ भारत हमारी जिम्मेदारी है।',
    2: 'मीडिया रिपोर्ट के अनुसार सरकार ने एक गुप्त योजना बनाई है जिसके तहत सभी नागरिकों की जासूसी की जाएगी, यह खबर सोशल मीडिया पर वायरल हो रही है।',
    3: 'भारतीय वैज्ञानिकों ने एक नई दवा की खोज की है जो कैंसर को तीन दिनों में ठीक कर सकती है और इसे जल्द ही बाजार में उपलब्ध कराया जाएगा।'
};

function loadSample(num) {
    const ta = document.getElementById('text-input');
    if (!ta) return;
    ta.value = SAMPLES[num] || '';
    // trigger char count update
    ta.dispatchEvent(new Event('input'));
    // scroll to textarea
    ta.focus();
}

// ----------------------------------------
// 3. PREDICT API CALL
// ----------------------------------------
async function predictText() {
    const ta         = document.getElementById('text-input');
    const analyzeBtn = document.getElementById('analyze-btn');
    const loading    = document.getElementById('loading');
    const results    = document.getElementById('results');
    const errorBox   = document.getElementById('error-box');

    // Reset
    errorBox.classList.add('hidden');
    results.classList.add('hidden');

    const inputText = ta ? ta.value.trim() : '';

    if (!inputText) {
        showError('Please enter some Hindi text before analyzing.', errorBox);
        return;
    }

    if (inputText.length < 5) {
        showError('Text is too short for meaningful analysis. Please enter a longer passage.', errorBox);
        return;
    }

    analyzeBtn.disabled = true;
    loading.classList.remove('hidden');

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: inputText })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || `Server returned ${response.status}`);
        }

        renderResults(data, inputText);

    } catch (err) {
        showError(err.message || 'Failed to connect to the analysis engine. Please try again.', errorBox);
    } finally {
        analyzeBtn.disabled = false;
        loading.classList.add('hidden');
    }
}

// ----------------------------------------
// 4. RENDER RESULTS
// ----------------------------------------
function renderResults(data, inputText) {
    const resultsPanel = document.getElementById('results');
    const labelOut     = document.getElementById('label-out');
    const verdictIcon  = document.getElementById('verdict-icon');

    const legitFill = document.getElementById('legit-fill');
    const synthFill = document.getElementById('synth-fill');
    const legitVal  = document.getElementById('legit-val');
    const synthVal  = document.getElementById('synth-val');

    const metaInput   = document.getElementById('meta-input');
    const metaCleaned = document.getElementById('meta-cleaned');

    // ── Prediction Badge ──────────────────────────────
    const prediction = data.prediction || 'Unknown';
    const isLegit = prediction.toLowerCase() === 'legitimate';

    labelOut.textContent = prediction.toUpperCase();
    labelOut.className   = 'verdict-badge ' + (isLegit ? 'legit' : 'synth');

    // ── Verdict Icon ──────────────────────────────────
    if (verdictIcon) {
        verdictIcon.className = 'verdict-icon-wrap ' + (isLegit ? 'legit-icon' : 'synth-icon');
        verdictIcon.innerHTML = isLegit
            ? '<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/></svg>'
            : '<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="m15 9-6 6M9 9l6 6"/></svg>';
    }

    // ── Color theme for results panel ─────────────────
    resultsPanel.style.background    = isLegit ? 'rgba(16,185,129,0.05)' : 'rgba(239,68,68,0.05)';
    resultsPanel.style.borderColor   = isLegit ? 'rgba(16,185,129,0.25)' : 'rgba(239,68,68,0.25)';

    // ── Confidence Bars ───────────────────────────────
    const confs = data.confidence_matrix || {};
    const legitScore = ((confs['Legitimate'] || 0) * 100);
    const synthScore = ((confs['Synthetic']  || 0) * 100);

    // Reset to 0 first (for re-animation)
    if (legitFill) legitFill.style.width = '0%';
    if (synthFill) synthFill.style.width = '0%';
    if (legitVal)  legitVal.textContent  = '0%';
    if (synthVal)  synthVal.textContent  = '0%';

    // Animate after a short tick
    setTimeout(() => {
        if (legitFill) legitFill.style.width = `${legitScore.toFixed(1)}%`;
        if (synthFill) synthFill.style.width = `${synthScore.toFixed(1)}%`;
        if (legitVal)  legitVal.textContent  = `${legitScore.toFixed(1)}%`;
        if (synthVal)  synthVal.textContent  = `${synthScore.toFixed(1)}%`;
    }, 80);

    // ── Metadata ──────────────────────────────────────
    if (metaInput)   metaInput.textContent   = `"${(data.input_text || inputText).substring(0, 120)}${inputText.length > 120 ? '…' : ''}"`;
    if (metaCleaned) metaCleaned.textContent = `${inputText.toLowerCase().replace(/\s+/g, ' ').trim().substring(0, 60)}…`;

    // Show the panel
    resultsPanel.classList.remove('hidden');
    resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ----------------------------------------
// 5. ERROR HELPER
// ----------------------------------------
function showError(msg, box) {
    if (!box) return;
    box.textContent = `⚠ ${msg}`;
    box.classList.remove('hidden');
}

// ----------------------------------------
// 6. MODEL FILTER
// ----------------------------------------
function filterModels(category) {
    // Update button states
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.filter === category);
    });

    // Show/hide cards
    document.querySelectorAll('.model-card').forEach(card => {
        const cardCat = card.dataset.category;
        if (category === 'all' || cardCat === category) {
            card.classList.remove('hidden');
        } else {
            card.classList.add('hidden');
        }
    });
}

// ----------------------------------------
// 7. NAVBAR SCROLL EFFECT
// ----------------------------------------
const navbar = document.getElementById('navbar');
if (navbar) {
    window.addEventListener('scroll', () => {
        if (window.scrollY > 20) {
            navbar.style.boxShadow = '0 8px 32px rgba(0,0,0,0.5)';
        } else {
            navbar.style.boxShadow = 'none';
        }
    }, { passive: true });
}

// ----------------------------------------
// 8. KEYBOARD SHORTCUT: Ctrl + Enter to analyze
// ----------------------------------------
document.addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        const btn = document.getElementById('analyze-btn');
        if (btn && !btn.disabled) btn.click();
    }
});

// ----------------------------------------
// 9. INTERSECTION OBSERVER — animate sections
// ----------------------------------------
const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, { threshold: 0.08, rootMargin: '0px 0px -40px 0px' });

document.querySelectorAll('.model-card, .arch-node, .stack-layer').forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(18px)';
    el.style.transition = 'opacity 0.55s ease, transform 0.55s ease';
    observer.observe(el);
});
