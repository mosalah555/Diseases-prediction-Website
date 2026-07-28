/* ============================================================
   1. MODEL CONFIGURATION
   Each field has TWO names:
     - id:     a safe DOM element id (no spaces/parentheses), used
               only for <label for="...">/<input id="...">
     - column: the EXACT column name from the training CSV. This is
               the key sent to Flask in the JSON body, so
               pd.DataFrame([values])[FEATURE_ORDER] on the backend
               lines up with the dataset columns by NAME, not by
               position -- matching what you decided earlier.

   Fields are listed in the exact order of each dataset's columns
   (target/label column excluded).

   Scoring happens server-side in Flask (see app.py ->
   POST /api/predict/<model_id>); this file only builds the form
   and displays whatever the API returns.
   ============================================================ */

const MODELS = [
  {
    id:'heart',
    label:'Heart Attack',
    code:'CARD-01',
    desc:'Cardiovascular event risk from vitals, labs, and lifestyle factors.',
    source:'disease_prediction__edited.csv',
    fields:[
      {id:'age', column:'age', label:'Age', unit:'years', type:'number', placeholder:'32', sample:32},
      {id:'gender', column:'gender', label:'Gender', type:'select', options:[['1','Male'],['0','Female']], sample:'1'},
      {id:'glucose_mg_dl', column:'glucose_mg_dl', label:'Glucose', unit:'mg/dL', type:'number', placeholder:'101', sample:101},
      {id:'cholesterol_mg_dl', column:'cholesterol_mg_dl', label:'Cholesterol', unit:'mg/dL', type:'number', placeholder:'235', sample:235},
      {id:'systolic_bp', column:'systolic_bp', label:'Systolic BP', unit:'mmHg', type:'number', placeholder:'152', sample:152},
      {id:'diastolic_bp', column:'diastolic_bp', label:'Diastolic BP', unit:'mmHg', type:'number', placeholder:'79', sample:79},
      {id:'heart_rate', column:'heart_rate', label:'Heart Rate', unit:'bpm', type:'number', placeholder:'73', sample:73},
      {id:'alcohol_consumption', column:'alcohol_consumption', label:'Alcohol Consumption', type:'select', options:[['1','Yes'],['0','No']], sample:'1'},
      {id:'smoking', column:'smoking', label:'Smoking', type:'select', options:[['1','Yes'],['0','No']], sample:'0'},
      {id:'bmi', column:'bmi', label:'BMI', unit:'kg/m²', type:'number', placeholder:'28.5', step:'0.1', sample:28.5},
      {id:'physical_activity', column:'physical_activity', label:'Physical Activity', type:'select', options:[['0','Low'],['1','Moderate'],['2','High']], sample:'0'},
      {id:'family_history', column:'family_history', label:'Family History', type:'select', options:[['1','Yes'],['0','No']], sample:'1'},
    ],
  },
  {
    id:'anemia',
    label:'Anemia',
    code:'HEMA-02',
    desc:'9-class blood disorder screen from a full CBC (complete blood count) panel.',
    source:'diagnosed_cbc_data_v4.csv',
    fields:[
      {id:'WBC', column:'WBC', label:'WBC', unit:'x10³/µL', type:'number', placeholder:'10.0', step:'0.1', sample:10.0},
      {id:'LYMp', column:'LYMp', label:'Lymphocyte %', unit:'%', type:'number', placeholder:'43.2', step:'0.1', sample:43.2},
      {id:'NEUTp', column:'NEUTp', label:'Neutrophil %', unit:'%', type:'number', placeholder:'50.1', step:'0.1', sample:50.1},
      {id:'LYMn', column:'LYMn', label:'Lymphocyte Count', unit:'x10³/µL', type:'number', placeholder:'4.3', step:'0.1', sample:4.3},
      {id:'NEUTn', column:'NEUTn', label:'Neutrophil Count', unit:'x10³/µL', type:'number', placeholder:'5.0', step:'0.1', sample:5.0},
      {id:'RBC', column:'RBC', label:'RBC', unit:'x10⁶/µL', type:'number', placeholder:'2.77', step:'0.01', sample:2.77},
      {id:'HGB', column:'HGB', label:'Hemoglobin', unit:'g/dL', type:'number', placeholder:'7.3', step:'0.1', sample:7.3},
      {id:'HCT', column:'HCT', label:'Hematocrit', unit:'%', type:'number', placeholder:'24.2', step:'0.1', sample:24.2},
      {id:'MCV', column:'MCV', label:'MCV', unit:'fL', type:'number', placeholder:'87.7', step:'0.1', sample:87.7},
      {id:'MCH', column:'MCH', label:'MCH', unit:'pg', type:'number', placeholder:'26.3', step:'0.1', sample:26.3},
      {id:'MCHC', column:'MCHC', label:'MCHC', unit:'g/dL', type:'number', placeholder:'30.1', step:'0.1', sample:30.1},
      {id:'PLT', column:'PLT', label:'Platelets', unit:'x10³/µL', type:'number', placeholder:'189', sample:189},
      {id:'PDW', column:'PDW', label:'PDW', unit:'fL', type:'number', placeholder:'12.5', step:'0.1', sample:12.5},
      {id:'PCT', column:'PCT', label:'PCT', unit:'%', type:'number', placeholder:'0.17', step:'0.01', sample:0.17},
    ],
  },
  {
    id:'stroke',
    label:'Stroke',
    code:'NEURO-03',
    desc:'Ischemic stroke likelihood from demographic and vascular history.',
    source:'healthcare-dataset-stroke-data.csv',
    fields:[
      {id:'gender', column:'gender', label:'Gender', type:'select', options:[['Male','Male'],['Female','Female'],['Other','Other']], sample:'Male'},
      {id:'age', column:'age', label:'Age', unit:'years', type:'number', placeholder:'67', sample:67},
      {id:'hypertension', column:'hypertension', label:'Hypertension', type:'select', options:[['1','Yes'],['0','No']], sample:'0'},
      {id:'heart_disease', column:'heart_disease', label:'Heart Disease', type:'select', options:[['1','Yes'],['0','No']], sample:'1'},
      {id:'ever_married', column:'ever_married', label:'Ever Married', type:'select', options:[['Yes','Yes'],['No','No']], sample:'Yes'},
      {id:'work_type', column:'work_type', label:'Work Type', type:'select', options:[['Private','Private'],['Self-employed','Self-employed'],['Govt_job','Govt job'],['children','Child'],['Never_worked','Never worked']], sample:'Private'},
      {id:'Residence_type', column:'Residence_type', label:'Residence Type', type:'select', options:[['Urban','Urban'],['Rural','Rural']], sample:'Urban'},
      {id:'avg_glucose_level', column:'avg_glucose_level', label:'Avg Glucose', unit:'mg/dL', type:'number', placeholder:'228.69', step:'0.01', sample:228.69},
      {id:'bmi', column:'bmi', label:'BMI', unit:'kg/m²', type:'number', placeholder:'36.6', step:'0.1', sample:36.6},
      {id:'smoking_status', column:'smoking_status', label:'Smoking Status', type:'select', options:[['formerly smoked','Formerly Smoked'],['never smoked','Never Smoked'],['smokes','Currently Smokes'],['Unknown','Unknown']], sample:'formerly smoked'},
    ],
  },
  {
    id:'kidney',
    label:'Kidney Disease',
    code:'RENAL-04',
    desc:'Chronic kidney disease risk tier from urinalysis, blood chemistry, and history.',
    source:'kidney_disease_dataset.csv',
    fields:[
      {id:'kd_age', column:'Age of the patient', label:'Age', unit:'years', type:'number', placeholder:'54', sample:54},
      {id:'kd_bp', column:'Blood pressure (mm/Hg)', label:'Blood Pressure', unit:'mmHg', type:'number', placeholder:'167', sample:167},
      {id:'kd_sg', column:'Specific gravity of urine', label:'Specific Gravity (Urine)', type:'number', placeholder:'1.023', step:'0.001', sample:1.023},
      {id:'kd_albumin', column:'Albumin in urine', label:'Albumin (Urine)', unit:'0–5', type:'number', placeholder:'1', sample:1},
      {id:'kd_sugar', column:'Sugar in urine', label:'Sugar (Urine)', unit:'0–5', type:'number', placeholder:'4', sample:4},
      {id:'kd_rbc', column:'Red blood cells in urine', label:'RBC (Urine)', type:'select', options:[['normal','Normal'],['abnormal','Abnormal']], sample:'normal'},
      {id:'kd_pus_cells', column:'Pus cells in urine', label:'Pus Cells (Urine)', type:'select', options:[['normal','Normal'],['abnormal','Abnormal']], sample:'abnormal'},
      {id:'kd_pus_clumps', column:'Pus cell clumps in urine', label:'Pus Cell Clumps', type:'select', options:[['not present','Not Present'],['present','Present']], sample:'not present'},
      {id:'kd_bacteria', column:'Bacteria in urine', label:'Bacteria (Urine)', type:'select', options:[['not present','Not Present'],['present','Present']], sample:'not present'},
      {id:'kd_glucose', column:'Random blood glucose level (mg/dl)', label:'Random Blood Glucose', unit:'mg/dL', type:'number', placeholder:'96', sample:96},
      {id:'kd_urea', column:'Blood urea (mg/dl)', label:'Blood Urea', unit:'mg/dL', type:'number', placeholder:'169.1', step:'0.1', sample:169.1},
      {id:'kd_creatinine', column:'Serum creatinine (mg/dl)', label:'Serum Creatinine', unit:'mg/dL', type:'number', placeholder:'7.55', step:'0.01', sample:7.55},
      {id:'kd_sodium', column:'Sodium level (mEq/L)', label:'Sodium', unit:'mEq/L', type:'number', placeholder:'146.1', step:'0.1', sample:146.1},
      {id:'kd_potassium', column:'Potassium level (mEq/L)', label:'Potassium', unit:'mEq/L', type:'number', placeholder:'6.27', step:'0.01', sample:6.27},
      {id:'kd_hemoglobin', column:'Hemoglobin level (gms)', label:'Hemoglobin', unit:'g', type:'number', placeholder:'11.8', step:'0.1', sample:11.8},
      {id:'kd_pcv', column:'Packed cell volume (%)', label:'Packed Cell Volume', unit:'%', type:'number', placeholder:'35', sample:35},
      {id:'kd_wbc', column:'White blood cell count (cells/cumm)', label:'WBC Count', unit:'cells/cumm', type:'number', placeholder:'5791', sample:5791},
      {id:'kd_rbc_count', column:'Red blood cell count (millions/cumm)', label:'RBC Count', unit:'M/cumm', type:'number', placeholder:'5.6', step:'0.1', sample:5.6},
      {id:'kd_hypertension', column:'Hypertension (yes/no)', label:'Hypertension', type:'select', options:[['yes','Yes'],['no','No']], sample:'yes'},
      {id:'kd_diabetes', column:'Diabetes mellitus (yes/no)', label:'Diabetes Mellitus', type:'select', options:[['yes','Yes'],['no','No']], sample:'yes'},
      {id:'kd_cad', column:'Coronary artery disease (yes/no)', label:'Coronary Artery Disease', type:'select', options:[['yes','Yes'],['no','No']], sample:'no'},
      {id:'kd_appetite', column:'Appetite (good/poor)', label:'Appetite', type:'select', options:[['good','Good'],['poor','Poor']], sample:'good'},
      {id:'kd_pedal_edema', column:'Pedal edema (yes/no)', label:'Pedal Edema', type:'select', options:[['yes','Yes'],['no','No']], sample:'no'},
      {id:'kd_anemia', column:'Anemia (yes/no)', label:'Anemia', type:'select', options:[['yes','Yes'],['no','No']], sample:'no'},
      {id:'kd_egfr', column:'Estimated Glomerular Filtration Rate (eGFR)', label:'eGFR', unit:'mL/min', type:'number', placeholder:'71.62', step:'0.01', sample:71.62},
      {id:'kd_upcr', column:'Urine protein-to-creatinine ratio', label:'Urine Protein/Creatinine', type:'number', placeholder:'2.51', step:'0.01', sample:2.51},
      {id:'kd_urine_output', column:'Urine output (ml/day)', label:'Urine Output', unit:'mL/day', type:'number', placeholder:'1397', sample:1397},
      {id:'kd_serum_albumin', column:'Serum albumin level', label:'Serum Albumin', unit:'g/dL', type:'number', placeholder:'3.23', step:'0.01', sample:3.23},
      {id:'kd_cholesterol', column:'Cholesterol level', label:'Cholesterol', unit:'mg/dL', type:'number', placeholder:'152', sample:152},
      {id:'kd_pth', column:'Parathyroid hormone (PTH) level', label:'Parathyroid Hormone (PTH)', unit:'pg/mL', type:'number', placeholder:'65.08', step:'0.01', sample:65.08},
      {id:'kd_calcium', column:'Serum calcium level', label:'Serum Calcium', unit:'mg/dL', type:'number', placeholder:'8.71', step:'0.01', sample:8.71},
      {id:'kd_phosphate', column:'Serum phosphate level', label:'Serum Phosphate', unit:'mg/dL', type:'number', placeholder:'4.31', step:'0.01', sample:4.31},
      {id:'kd_family_history', column:'Family history of chronic kidney disease', label:'Family History of CKD', type:'select', options:[['yes','Yes'],['no','No']], sample:'no'},
      {id:'kd_smoking', column:'Smoking status', label:'Smoking Status', type:'select', options:[['yes','Yes'],['no','No']], sample:'yes'},
      {id:'kd_bmi', column:'Body Mass Index (BMI)', label:'BMI', unit:'kg/m²', type:'number', placeholder:'25.3', step:'0.1', sample:25.3},
      {id:'kd_physical_activity', column:'Physical activity level', label:'Physical Activity Level', type:'select', options:[['low','Low'],['moderate','Moderate'],['high','High']], sample:'low'},
      {id:'kd_diabetes_duration', column:'Duration of diabetes mellitus (years)', label:'Diabetes Duration', unit:'years', type:'number', placeholder:'4', sample:4},
      {id:'kd_hypertension_duration', column:'Duration of hypertension (years)', label:'Hypertension Duration', unit:'years', type:'number', placeholder:'16', sample:16},
      {id:'kd_cystatin_c', column:'Cystatin C level', label:'Cystatin C', unit:'mg/L', type:'number', placeholder:'0.67', step:'0.01', sample:0.67},
      {id:'kd_sediment', column:'Urinary sediment microscopy results', label:'Urinary Sediment Microscopy', type:'select', options:[['normal','Normal'],['abnormal','Abnormal']], sample:'normal'},
      {id:'kd_crp', column:'C-reactive protein (CRP) level', label:'C-Reactive Protein (CRP)', unit:'mg/L', type:'number', placeholder:'4.88', step:'0.01', sample:4.88},
      {id:'kd_il6', column:'Interleukin-6 (IL-6) level', label:'Interleukin-6 (IL-6)', unit:'pg/mL', type:'number', placeholder:'10.23', step:'0.01', sample:10.23},
    ],
  },
  {
    id:'diabetes',
    label:'Diabetes',
    code:'ENDO-05',
    desc:'Type 2 diabetes likelihood from glycemic markers and health history.',
    source:'diabetes_prediction_dataset.csv',
    fields:[
      {id:'gender', column:'gender', label:'Gender', type:'select', options:[['Female','Female'],['Male','Male'],['Other','Other']], sample:'Female'},
      {id:'age', column:'age', label:'Age', unit:'years', type:'number', placeholder:'80', sample:80},
      {id:'hypertension', column:'hypertension', label:'Hypertension', type:'select', options:[['1','Yes'],['0','No']], sample:'0'},
      {id:'heart_disease', column:'heart_disease', label:'Heart Disease', type:'select', options:[['1','Yes'],['0','No']], sample:'1'},
      {id:'smoking_history', column:'smoking_history', label:'Smoking History', type:'select', options:[['never','Never'],['No Info','No Info'],['current','Current'],['former','Former'],['ever','Ever'],['not current','Not Current']], sample:'never'},
      {id:'bmi', column:'bmi', label:'BMI', unit:'kg/m²', type:'number', placeholder:'25.19', step:'0.01', sample:25.19},
      {id:'HbA1c_level', column:'HbA1c_level', label:'HbA1c Level', unit:'%', type:'number', placeholder:'6.6', step:'0.1', sample:6.6},
      {id:'blood_glucose_level', column:'blood_glucose_level', label:'Blood Glucose', unit:'mg/dL', type:'number', placeholder:'140', sample:140},
    ],
  },
  {
    id:'liver',
    label:'Liver Disease',
    code:'HEPA-06',
    desc:'Hepatic function screen from bilirubin, enzyme, and protein levels.',
    source:'Indian_Liver_Patient_Dataset__ILPD_.csv',
    fields:[
      {id:'age', column:'age', label:'Age', unit:'years', type:'number', placeholder:'65', sample:65},
      {id:'gender', column:'gender', label:'Gender', type:'select', options:[['Female','Female'],['Male','Male']], sample:'Female'},
      {id:'tot_bilirubin', column:'tot_bilirubin', label:'Total Bilirubin', unit:'mg/dL', type:'number', placeholder:'0.7', step:'0.1', sample:0.7},
      {id:'direct_bilirubin', column:'direct_bilirubin', label:'Direct Bilirubin', unit:'mg/dL', type:'number', placeholder:'0.1', step:'0.1', sample:0.1},
      {id:'tot_proteins', column:'tot_proteins', label:'Total Proteins', unit:'IU/L', type:'number', placeholder:'187', sample:187},
      {id:'albumin', column:'albumin', label:'Albumin', unit:'IU/L', type:'number', placeholder:'16', sample:16},
      {id:'ag_ratio', column:'ag_ratio', label:'A/G Ratio', type:'number', placeholder:'18', sample:18},
      {id:'sgpt', column:'sgpt', label:'SGPT (ALT)', unit:'IU/L', type:'number', placeholder:'6.8', step:'0.1', sample:6.8},
      {id:'sgot', column:'sgot', label:'SGOT (AST)', unit:'IU/L', type:'number', placeholder:'3.3', step:'0.1', sample:3.3},
      {id:'alkphos', column:'alkphos', label:'Alk. Phosphatase', unit:'IU/L', type:'number', placeholder:'0.9', step:'0.01', sample:0.9},
    ],
  },
];

/* ============================================================
   2. RENDERING
   ============================================================ */

let activeId = MODELS[0].id;

const tabsEl = document.getElementById('tabs');
const bodyEl = document.getElementById('chartBody');

function renderTabs(){
  tabsEl.innerHTML = MODELS.map((m, i) => `
    <button class="tab ${m.id===activeId?'active':''}" data-id="${m.id}">
      <span class="num">${String(i+1).padStart(2,'0')}</span>${m.label}
    </button>
  `).join('');
  tabsEl.querySelectorAll('.tab').forEach(btn=>{
    btn.addEventListener('click', ()=>{
      activeId = btn.dataset.id;
      renderTabs();
      renderChart();
    });
  });
}

function fieldHtml(f){
  if(f.type === 'select'){
    return `
      <div class="field">
        <label for="${f.id}">${f.label}</label>
        <select id="${f.id}">
          ${f.options.map(([val,text])=>`<option value="${val}">${text}</option>`).join('')}
        </select>
      </div>`;
  }
  return `
    <div class="field">
      <label for="${f.id}">${f.label} ${f.unit ? `<span class="unit">${f.unit}</span>` : ''}</label>
      <input id="${f.id}" type="number" ${f.step?`step="${f.step}"`:''} placeholder="${f.placeholder||''}">
    </div>`;
}

function renderChart(){
  const model = MODELS.find(m=>m.id===activeId);
  bodyEl.innerHTML = `
    <div class="chart-head">
      <div>
        <h2>${model.label}</h2>
        <p>${model.desc}</p>
      </div>
      <div class="req-code">MODEL ${model.code}</div>
    </div>
    <form id="riskForm" autocomplete="off">
      ${model.fields.map(fieldHtml).join('')}
      <div class="actions">
        <button type="submit" class="run"><span class="bolt">▸</span> Run Model</button>
        <button type="button" class="reset" id="resetBtn">Clear Fields</button>
        <button type="button" class="sample-fill" id="sampleBtn">Fill sample values</button>
      </div>
      <div class="readout" id="readout"></div>
    </form>
  `;

  document.getElementById('sampleBtn').addEventListener('click', ()=>{
    model.fields.forEach(f=>{
      const el = document.getElementById(f.id);
      if(el) el.value = f.sample;
    });
  });

  document.getElementById('resetBtn').addEventListener('click', ()=>{
    document.getElementById('riskForm').reset();
    document.getElementById('readout').className = 'readout';
    document.getElementById('readout').innerHTML = '';
  });

  document.getElementById('riskForm').addEventListener('submit', (e)=>{
    e.preventDefault();
    runModel(model);
  });
}

async function runModel(model){
  // IMPORTANT: keys here are f.column (the exact dataset column name),
  // NOT f.id (which is just a safe DOM id). This is what makes the
  // JSON body match the training CSV columns 1:1 on the backend.
  const values = {};
  model.fields.forEach(f=>{
    const el = document.getElementById(f.id);
    values[f.column] = f.type === 'select' ? el.value : parseFloat(el.value);
  });

  const submitBtn = document.querySelector('#riskForm button.run');
  const readout = document.getElementById('readout');

  submitBtn.disabled = true;
  submitBtn.textContent = 'Running…';

  try{
    const res = await fetch(`/api/predict/${model.id}`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(values)
    });

    if(!res.ok){
      const errBody = await res.json().catch(()=>({}));
      throw new Error(errBody.error || `Request failed (${res.status})`);
    }

    const { riskScore, level, levelText, detail, predictedLabel, message, recommendation, confidencePct } = await res.json();

    readout.className = `readout show level-${level}`;
    readout.innerHTML = `
      <div class="rd-left">
         <div class="rd-label">${model.label} — Model Output</div>
         <p class="rd-level">${predictedLabel ? predictedLabel : levelText}</p>
         ${predictedLabel ? `<p class="rd-badge">${levelText}</p>` : ''}
         <p class="rd-detail">${message || levelText}</p>
         ${detail ? `<p class="rd-detail" style="opacity:0.8;">${detail}</p>` : ''}
         ${recommendation ? `<p class="rd-detail"><strong>Recommendation:</strong> ${recommendation}</p>` : ''}
         ${confidencePct ? `<p class="rd-detail" style="opacity:0.7; font-size:0.9em;">Model confidence: ${confidencePct}%</p>` : ''}
      </div>
      <div class="gauge">
         <div class="score">${Math.round(riskScore)}</div>
         <div class="of">/ 100</div>
         ${gaugeSvg(riskScore, level)}
      </div>
    `;
  } catch(err){
    readout.className = 'readout show';
    readout.innerHTML = `
      <div class="rd-left">
        <div class="rd-label">${model.label} — Request Failed</div>
        <p class="rd-error">${err.message}</p>
      </div>
    `;
  } finally {
    submitBtn.disabled = false;
    submitBtn.innerHTML = '<span class="bolt">▸</span> Run Model';
    readout.scrollIntoView({behavior:'smooth', block:'nearest'});
  }
}

function gaugeSvg(score, level){
  const colors = { low:'#7FD6A0', moderate:'#E7B65C', high:'#E88A72' };
  const r = 34, c = 2*Math.PI*r;
  const offset = c - (c * score/100);
  return `
    <svg width="90" height="90" viewBox="0 0 90 90">
      <circle cx="45" cy="45" r="${r}" fill="none" stroke="#2A3E56" stroke-width="7"/>
      <circle cx="45" cy="45" r="${r}" fill="none" stroke="${colors[level] || '#7FD6A0'}" stroke-width="7"
        stroke-linecap="round" stroke-dasharray="${c}" stroke-dashoffset="${offset}"
        transform="rotate(-90 45 45)" style="transition: stroke-dashoffset .6s ease;"/>
    </svg>
  `;
}

renderTabs();
renderChart();