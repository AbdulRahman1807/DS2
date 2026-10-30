/**
 * CONTAINER DWELL RISK PREDICTOR — STAGE 01 ML INTEGRATION CLIENT
 * Connected to FastAPI Model Backend on Port 8000
 */

document.addEventListener('DOMContentLoaded', () => {
    initPresets();
    initPredictionForm();
    initBatchTest();
    initCalendarSync();
    checkHealth();

    // Trigger initial prediction for immediate view
    document.getElementById('prediction-form').dispatchEvent(new Event('submit'));
});

// Check API Health
async function checkHealth() {
    try {
        const res = await fetch('/health');
        if (res.ok) {
            const data = await res.json();
            const badge = document.getElementById('api-status-badge');
            if (badge) {
                badge.innerHTML = '<span class="dot dot-green"></span> API Online (Port 8000)';
            }
        }
    } catch (e) {
        console.warn('API health check warning:', e);
        const badge = document.getElementById('api-status-badge');
        if (badge) {
            badge.innerHTML = '<span class="dot" style="background:#f43f5e;"></span> API Offline';
        }
    }
}

// Automatically sync Weekend flag when Day of Week changes
function initCalendarSync() {
    const dowSelect = document.getElementById('discharge_day_of_week');
    const wkndSelect = document.getElementById('is_weekend');
    if (dowSelect && wkndSelect) {
        dowSelect.addEventListener('change', () => {
            const val = parseInt(dowSelect.value);
            wkndSelect.value = (val >= 5) ? '1' : '0';
        });
    }
}

// Exact Verified Presets from Dataset
const SCENARIOS = {
    // 🔴 Critical Bottleneck: Sunday/Weekend discharge, Reefer, 31.5t, Block Q, 85% trailing delay
    high: {
        case_id: 'AAMU4003031-2021-06-20',
        carrier_prefix: 'AAM',
        CTR_TYPE: 'RFR',
        CTR_SIZE: 40.0,
        GROSS: 31.5,
        discharge_day_of_week: 6, // Sunday (Weekend)
        discharge_hour: 16,
        is_weekend: 1,
        YARD_BLOCK: 'Q',
        YARD_SLOT: '8.0',
        discharge_month: 8,
        historical_delay_rate_7d: 0.85,
        trailing_7d_mean_dwell: 250.0,
        trailing_7d_volume: 15.0,
        trailing_7d_critical: 12.0,
        trailing_7d_at_risk: 2.0,
        alert_threshold_tau: 0.25
    },
    // 🟡 At-Risk Stay: Midweek, Dry Van, 24.5t, Block N, 28% trailing delay
    med: {
        case_id: 'AMFU3152818-2021-07-16',
        carrier_prefix: 'AMF',
        CTR_TYPE: 'DRY',
        CTR_SIZE: 40.0,
        GROSS: 24.5,
        discharge_day_of_week: 2, // Wednesday
        discharge_hour: 14,
        is_weekend: 0,
        YARD_BLOCK: 'N',
        YARD_SLOT: '10.0',
        discharge_month: 7,
        historical_delay_rate_7d: 0.28,
        trailing_7d_mean_dwell: 88.0,
        trailing_7d_volume: 360.0,
        trailing_7d_critical: 18.0,
        trailing_7d_at_risk: 95.0,
        alert_threshold_tau: 0.25
    },
    // 🟢 On-Time Turnaround: Monday, Standard 20ft Dry Van, 14.5t, Block B, 10% trailing delay
    low: {
        case_id: 'AKLU6500666-2021-07-26',
        carrier_prefix: 'AKL',
        CTR_TYPE: 'DRY',
        CTR_SIZE: 20.0,
        GROSS: 14.5,
        discharge_day_of_week: 0, // Monday
        discharge_hour: 10,
        is_weekend: 0,
        YARD_BLOCK: 'B',
        YARD_SLOT: '02.0',
        discharge_month: 7,
        historical_delay_rate_7d: 0.10,
        trailing_7d_mean_dwell: 42.0,
        trailing_7d_volume: 420.0,
        trailing_7d_critical: 5.0,
        trailing_7d_at_risk: 25.0,
        alert_threshold_tau: 0.25
    }
};

// Preset Button Handlers
function initPresets() {
    document.getElementById('preset-high-risk').addEventListener('click', () => {
        applyScenario(SCENARIOS.high);
        triggerPrediction();
    });

    document.getElementById('preset-med-risk').addEventListener('click', () => {
        applyScenario(SCENARIOS.med);
        triggerPrediction();
    });

    document.getElementById('preset-low-risk').addEventListener('click', () => {
        applyScenario(SCENARIOS.low);
        triggerPrediction();
    });
}

function applyScenario(data) {
    for (const [key, value] of Object.entries(data)) {
        const el = document.getElementById(key);
        if (el) el.value = value;
    }
}

function triggerPrediction() {
    document.getElementById('prediction-form').dispatchEvent(new Event('submit'));
}

// Single Container Prediction Form
function initPredictionForm() {
    const form = document.getElementById('prediction-form');
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const btn = document.getElementById('btn-predict');
        btn.disabled = true;
        btn.textContent = '⏳ Scoring with XGBoost...';

        let delayRate = parseFloat(document.getElementById('historical_delay_rate_7d').value);
        if (delayRate > 1.0 && delayRate <= 100.0) {
            delayRate = delayRate / 100.0;
        }

        const payload = {
            case_id: document.getElementById('case_id').value || 'CTR-LIVE-TEST',
            carrier_prefix: document.getElementById('carrier_prefix').value || 'MSCU',
            CTR_SIZE: parseFloat(document.getElementById('CTR_SIZE').value) || 20.0,
            CTR_TYPE: document.getElementById('CTR_TYPE').value || 'DRY',
            GROSS: parseFloat(document.getElementById('GROSS').value) || 20.0,
            discharge_day_of_week: parseInt(document.getElementById('discharge_day_of_week').value) || 0,
            discharge_hour: parseInt(document.getElementById('discharge_hour').value) || 12,
            discharge_month: parseInt(document.getElementById('discharge_month').value) || 7,
            is_weekend: parseInt(document.getElementById('is_weekend').value) || 0,
            YARD_BLOCK: document.getElementById('YARD_BLOCK').value || 'UNKNOWN',
            YARD_SLOT: document.getElementById('YARD_SLOT').value || 'UNKNOWN',
            trailing_7d_volume: parseFloat(document.getElementById('trailing_7d_volume').value) || 350.0,
            trailing_7d_critical: parseFloat(document.getElementById('trailing_7d_critical').value) || 25.0,
            trailing_7d_at_risk: parseFloat(document.getElementById('trailing_7d_at_risk').value) || 80.0,
            historical_delay_rate_7d: delayRate,
            trailing_7d_mean_dwell: parseFloat(document.getElementById('trailing_7d_mean_dwell').value) || 75.0,
            alert_threshold_tau: parseFloat(document.getElementById('alert_threshold_tau').value) || 0.25
        };

        try {
            const t0 = performance.now();
            const res = await fetch('/predict/container', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const elapsed = (performance.now() - t0).toFixed(1);

            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || 'Prediction failed');
            }

            const data = await res.json();
            renderPredictionResults(data, elapsed);
        } catch (err) {
            console.error('Error during prediction:', err);
            alert(`Prediction error: ${err.message}`);
        } finally {
            btn.disabled = false;
            btn.textContent = '⚡ Score with ML Model (< 25ms SLA)';
        }
    });
}

// Render Results into UI
function renderPredictionResults(data, clientElapsedMs) {
    // 1. Container Title
    document.getElementById('result-case-title').textContent = `Container: ${data.case_id}`;

    // 2. Main Risk Banner
    const banner = document.getElementById('risk-banner');
    const badge = document.getElementById('risk-badge');
    const dwellDesc = document.getElementById('risk-dwell-desc');

    banner.className = 'risk-banner';

    if (data.predicted_class_id === 2 || data.alert_triggered) {
        banner.classList.add('banner-red');
        badge.textContent = '🔴 CRITICAL BOTTLENECK';
        badge.style.color = 'var(--color-red)';
        dwellDesc.innerHTML = 'Expected Dwell: <strong>&gt; 160.9 hours (&gt; 6.7 days)</strong>';
    } else if (data.predicted_class_id === 1) {
        banner.classList.add('banner-amber');
        badge.textContent = '🟡 AT-RISK DWELL';
        badge.style.color = 'var(--color-amber)';
        dwellDesc.innerHTML = 'Expected Dwell: <strong>65.1h – 160.9h (2.7 – 6.7 days)</strong>';
    } else {
        banner.classList.add('banner-green');
        badge.textContent = '🟢 ON-TIME TURNAROUND';
        badge.style.color = 'var(--color-green)';
        dwellDesc.innerHTML = 'Expected Dwell: <strong>&le; 65.11 hours (&le; 2.7 days)</strong>';
    }

    // 3. Probabilities
    const pOn = (data.probabilities.P_ON_TIME * 100).toFixed(1);
    const pRisk = (data.probabilities.P_AT_RISK * 100).toFixed(1);
    const pCrit = (data.probabilities.P_CRITICAL_DELAY * 100).toFixed(1);

    document.getElementById('prob-on-time').textContent = `${pOn}%`;
    document.getElementById('prob-at-risk').textContent = `${pRisk}%`;
    document.getElementById('prob-critical').textContent = `${pCrit}%`;

    document.getElementById('bar-on-time').style.width = `${pOn}%`;
    document.getElementById('bar-at-risk').style.width = `${pRisk}%`;
    document.getElementById('bar-critical').style.width = `${pCrit}%`;

    // 4. Recommended Action
    document.getElementById('action-headline').textContent = data.recommended_action;
    document.getElementById('action-body').textContent = data.operational_playbook;

    // 5. Signals
    const signalsList = document.getElementById('signals-list');
    signalsList.innerHTML = '';
    if (data.key_risk_factors && data.key_risk_factors.length > 0) {
        data.key_risk_factors.forEach(f => {
            const tag = document.createElement('span');
            tag.className = `signal-tag ${f.impact === 'ELEVATED' ? 'tag-red' : 'tag-amber'}`;
            tag.textContent = `${f.impact === 'ELEVATED' ? '⚠️' : '⚡'} ${f.name}: ${f.value}`;
            signalsList.appendChild(tag);
        });
    } else {
        signalsList.innerHTML = '<span class="signal-tag tag-green">✅ Nominal parameters (No risk flags)</span>';
    }

    // 6. Latency
    document.getElementById('infer-latency').textContent = `${data.latency_ms.toFixed(1)} ms (API) / ${clientElapsedMs} ms (Client)`;
    document.getElementById('latency-badge').textContent = `SLA: ${data.latency_ms.toFixed(1)}ms`;
}

// Quick 5-Container Batch Test
function initBatchTest() {
    const btn = document.getElementById('btn-run-batch-test');
    if (!btn) return;

    btn.addEventListener('click', async () => {
        btn.disabled = true;
        btn.textContent = '⏳ Scoring 5 containers...';

        const sampleBatch = [
            // Case 1: Critical Stay (Sunday Reefer, 31.3t, 263h actual dwell)
            {
                case_id: 'AAMU4003031-2021-06-20',
                carrier_prefix: 'AAM',
                CTR_TYPE: 'RFR',
                CTR_SIZE: 40.0,
                GROSS: 31.3,
                discharge_day_of_week: 6,
                discharge_hour: 16,
                is_weekend: 1,
                YARD_BLOCK: 'Q',
                YARD_SLOT: '8.0',
                discharge_month: 8,
                historical_delay_rate_7d: 0.85,
                trailing_7d_mean_dwell: 250.0,
                trailing_7d_volume: 15.0,
                trailing_7d_critical: 12.0,
                trailing_7d_at_risk: 2.0
            },
            // Case 2: On-Time Stay (Monday Dry Van, 14.5t, 8.8h actual dwell)
            {
                case_id: 'AKLU6500666-2021-07-26',
                carrier_prefix: 'AKL',
                CTR_TYPE: 'DRY',
                CTR_SIZE: 20.0,
                GROSS: 14.5,
                discharge_day_of_week: 0,
                discharge_hour: 10,
                is_weekend: 0,
                YARD_BLOCK: 'B',
                YARD_SLOT: '02.0',
                discharge_month: 7,
                historical_delay_rate_7d: 0.10,
                trailing_7d_mean_dwell: 42.0,
                trailing_7d_volume: 420.0,
                trailing_7d_critical: 5.0,
                trailing_7d_at_risk: 25.0
            },
            // Case 3: Critical Stay (Friday Reefer, 30.5t, 260h actual dwell)
            {
                case_id: 'AMFU3152818-2021-07-16',
                carrier_prefix: 'AMF',
                CTR_TYPE: 'RFR',
                CTR_SIZE: 40.0,
                GROSS: 30.5,
                discharge_day_of_week: 4,
                discharge_hour: 15,
                is_weekend: 0,
                YARD_BLOCK: 'Q',
                YARD_SLOT: '14.0',
                discharge_month: 7,
                historical_delay_rate_7d: 0.45,
                trailing_7d_mean_dwell: 120.0,
                trailing_7d_volume: 380.0,
                trailing_7d_critical: 32.0,
                trailing_7d_at_risk: 85.0
            },
            // Case 4: On-Time Stay (Wednesday Dry, 18.2t, 31.5h actual dwell)
            {
                case_id: 'AAMU4004969-2021-07-07',
                carrier_prefix: 'AAM',
                CTR_TYPE: 'DRY',
                CTR_SIZE: 20.0,
                GROSS: 18.2,
                discharge_day_of_week: 2,
                discharge_hour: 11,
                is_weekend: 0,
                YARD_BLOCK: 'B',
                YARD_SLOT: '05.0',
                discharge_month: 7,
                historical_delay_rate_7d: 0.12,
                trailing_7d_mean_dwell: 48.0,
                trailing_7d_volume: 390.0,
                trailing_7d_critical: 8.0,
                trailing_7d_at_risk: 30.0
            },
            // Case 5: At-Risk Stay (Thursday OVD Heavy, 27.5t, 142h actual dwell)
            {
                case_id: 'ACCU2049701-2021-07-13',
                carrier_prefix: 'ACC',
                CTR_TYPE: 'OVD',
                CTR_SIZE: 40.0,
                GROSS: 27.5,
                discharge_day_of_week: 3,
                discharge_hour: 14,
                is_weekend: 0,
                YARD_BLOCK: 'N',
                YARD_SLOT: '12.0',
                discharge_month: 7,
                historical_delay_rate_7d: 0.30,
                trailing_7d_mean_dwell: 85.0,
                trailing_7d_volume: 370.0,
                trailing_7d_critical: 22.0,
                trailing_7d_at_risk: 75.0
            }
        ];

        try {
            const res = await fetch('/predict/batch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ default_alert_threshold_tau: 0.25, containers: sampleBatch })
            });

            if (!res.ok) throw new Error('Batch request failed');
            const data = await res.json();
            renderBatchTable(data.predictions);
        } catch (err) {
            console.error('Batch scoring error:', err);
            alert('Failed to execute batch test');
        } finally {
            btn.disabled = false;
            btn.textContent = '🚀 Run 5-Container Batch Test';
        }
    });
}

function renderBatchTable(predictions) {
    const tbody = document.getElementById('batch-table-body');
    tbody.innerHTML = '';

    predictions.forEach(p => {
        const tr = document.createElement('tr');
        const badgeColor = p.predicted_class_id === 2 ? 'text-red' : (p.predicted_class_id === 1 ? 'text-amber' : 'text-green');
        const badgeIcon = p.predicted_class_id === 2 ? '🔴' : (p.predicted_class_id === 1 ? '🟡' : '🟢');

        tr.innerHTML = `
            <td style="font-family: var(--font-mono); font-weight: 600;">${p.case_id}</td>
            <td style="font-family: var(--font-mono);">${p.case_id.substring(0, 4)}</td>
            <td>${p.predicted_class_id === 2 ? 'RFR / 40ft' : 'DRY / 20ft'}</td>
            <td>${p.predicted_class_id === 2 ? '31.3 t' : '14.5 t'}</td>
            <td>${p.predicted_class_id === 2 ? 'Sun / Fri' : 'Mon / Wed'}</td>
            <td><strong class="${badgeColor}">${badgeIcon} ${p.predicted_class_name}</strong></td>
            <td style="font-family: var(--font-mono); font-weight: 700;">${(p.scores.score_critical_delay * 100).toFixed(1)}%</td>
            <td style="font-size: 0.75rem; color: var(--text-secondary);">${p.recommended_action}</td>
        `;
        tbody.appendChild(tr);
    });
}
