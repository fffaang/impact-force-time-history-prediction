'use strict';
const $ = id => document.getElementById(id);
const state = {meta:null,current:null,comparisons:[],normalized:false,batch:null,busy:false,dirty:false};
const specs = [
  ['Fmax','Peak force','F<sub>max</sub>','kN',2],
  ['Td','Duration','T<sub>d</sub>','ms',2],
  ['I','Impulse','I','kN·ms',2],
  ['Astar','Pulse fullness','A<sup>*</sup>','',3],
  ['tp_ratio','Peak phase','t<sub>p</sub>/T<sub>d</sub>','',3]
];
const colors = ['#136e68','#bd4963','#3d74b8','#aa7523','#7461a5','#4f8274'];
const fmt = (n,d=3) => Number(n).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d,useGrouping:false});
const safeName = s => String(s).replace(/[<>:"/\\|?*\x00-\x1f]/g,'_').slice(0,90)||'prediction';
const icons = () => window.lucide?.createIcons();
function toast(text) { $('toast').textContent=text;$('toast').hidden=false;clearTimeout(toast.timer);toast.timer=setTimeout(()=>$('toast').hidden=true,3200); }
function download(text,name,type='text/csv;charset=utf-8') {
  const blob=new Blob([type.startsWith('text/csv')?'\ufeff':'',text],{type});
  const a=document.createElement('a'); a.href=URL.createObjectURL(blob);a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);
}
async function request(path,body,raw=false) {
  const options=body===undefined?{}:{method:'POST',headers:{'X-Session-Token':state.meta.token,'Content-Type':raw?'text/csv;charset=utf-8':'application/json'},body:raw?body:JSON.stringify(body)};
  const response=await fetch(path,options),result=await response.json();
  if(!response.ok){const error=new Error(result.error||'Request failed');error.details=result.details||[];throw error;}
  return result;
}
function fields() {
  const names={impact:'Impact & loading',geometry:'Member geometry',material:'Material properties',reinforcement:'Reinforcement'};
  for(const [group,label] of Object.entries(names)) {
    const section=document.createElement('fieldset');section.className='field-group';
    const title=document.createElement('legend');title.textContent=label;section.append(title);
    const grid=document.createElement('div');grid.className='fields';section.append(grid);
    for(const f of state.meta.fields.filter(f=>f.group===group)) {
      const item=document.createElement('div');item.className='field';
      const lab=document.createElement('label');lab.htmlFor='field-'+f.key;
      lab.innerHTML=`<span class="field-symbol">${f.symbol}</span><span>${f.label}${f.required?'<b class="required-mark">*</b>':''}</span>`;
      const wrap=document.createElement('div');wrap.className='number-wrap';
      const input=document.createElement('input');input.id='field-'+f.key;input.name=f.key;input.type='number';input.step='any';input.min=f.required||['d','H0'].includes(f.key)?'0.000000001':'0';input.required=!!f.required;input.inputMode='decimal';input.autocomplete='off';
      if(f.key==='corrosion_pct') input.max='100';
      input.placeholder=f.required?'Required':'Not provided';
      const range=state.meta.ranges[f.key];const rangeNumber=n=>new Intl.NumberFormat('en-US',{maximumSignificantDigits:5}).format(n);input.title=`${f.label}. Training range: ${rangeNumber(range.min)}–${rangeNumber(range.max)} ${f.unit}`;
      const unit=document.createElement('span');unit.className='unit';unit.textContent=f.unit;wrap.append(input,unit);item.append(lab,wrap);grid.append(item);
      input.addEventListener('input',()=>{input.classList.remove('invalid');$('example-select').value='';stale();});
      input.addEventListener('invalid',()=>{input.setCustomValidity(input.validity.valueMissing?`${f.label} is required.`:`Enter a valid ${f.label.toLowerCase()} in ${f.unit}.`);});
      input.addEventListener('input',()=>input.setCustomValidity(''));
    }
    $('field-groups').append(section);
  }
  specs.forEach(([key,label,symbol,unit])=>{
    const cell=document.createElement('div');cell.className='anchor';cell.innerHTML=`<div class="anchor-title"><span class="math">${symbol}</span><span>${label}</span></div><p class="anchor-value" id="anchor-${key}">—</p><span class="anchor-unit">${unit||'Dimensionless'}${key==='Astar'?' · Derived':''}</span>`;$('anchor-grid').append(cell);
  });
}
function readInputs() { return Object.fromEntries(state.meta.fields.map(f=>[f.key,$('field-'+f.key).value===''?null:Number($('field-'+f.key).value)])); }
function fill(inputs,name) { for(const f of state.meta.fields){$('field-'+f.key).value=inputs[f.key]??'';$('field-'+f.key).setCustomValidity('');}$('case-name').value=name||'New case';clearError();stale(); }
function stale() { state.dirty=true;$('prediction-state').textContent=state.current?'Inputs changed':'Ready to predict';$('prediction-state').classList.remove('ready');$('input-status').textContent='17 engineering inputs';exportEnabled(false); }
function exportEnabled(enabled) { for(const id of ['export-csv','export-json','export-png','export-svg','compare-add']) $(id).disabled=!enabled; }
function clearError() {$('form-error').hidden=true;document.querySelectorAll('input.invalid').forEach(x=>x.classList.remove('invalid'));}
function showError(error,id='form-error') {
  const texts=[error.message,...(error.details||[]).map(x=>x.row?`Row ${x.row}: ${(x.details||[]).map(d=>d.message).join('; ')||x.message}`:x.message)];
  $(id).textContent=texts.join('\n');$(id).hidden=false;
  for(const d of error.details||[]) if(d.field&&$('field-'+d.field)) $('field-'+d.field).classList.add('invalid');
}
function busy(on) {state.busy=on;$('chart-loading').hidden=!on;$('predict-button').disabled=on;$('predict-button').querySelector('span').textContent=on?'Computing':'Predict history';$('input-status').textContent=on?'Running frozen model':'17 engineering inputs';}
async function predict() {
  clearError();if(!$('parameter-form').reportValidity()||state.busy)return;
  busy(true);exportEnabled(false);
  try {await present(await request('/api/predict',{name:$('case-name').value.trim()||'New case',inputs:readInputs()}));}
  catch(error){showError(error);$('prediction-state').textContent='Prediction failed';}
  finally{busy(false);}
}
function switchTab(tab) { $('single-view').hidden=tab!=='single';$('batch-view').hidden=tab!=='batch';document.querySelectorAll('.tab').forEach(b=>b.classList.toggle('active',b.dataset.tab===tab));if(tab==='single')setTimeout(()=>Plotly.Plots.resize($('curve-chart')),0); }
async function present(result) {
  state.current=result;state.dirty=false;$('prediction-name').textContent=result.name;$('prediction-state').textContent='Prediction complete';$('prediction-state').classList.add('ready');
  for(const [key,,,unit,d] of specs) $('anchor-'+key).textContent=fmt(result.anchors[key],d);
  $('elapsed').textContent=`Inference ${fmt(result.elapsed_ms,0)} ms`;$('point-count').textContent=`${result.curve.time_ms.length} native nodes`;
  $('warnings').replaceChildren();
  result.warnings.forEach(w=>{const el=document.createElement('div');el.className='warning-row '+w.level;el.textContent=w.message;$('warnings').append(el);});
  $('consistency-table').replaceChildren();
  specs.forEach(([key,label,symbol,unit,d])=>{const tr=document.createElement('tr');const td=document.createElement('td');td.innerHTML=`<span class="math">${symbol}</span>`;tr.append(td);for(const val of [fmt(result.anchors[key],d+2),fmt(result.reconstructed[key],d+2),unit||'—']){const c=document.createElement('td');c.textContent=val;tr.append(c);}$('consistency-table').append(tr);});
  $('consistency-status').textContent=`Area residual ${Number(result.diagnostics.area_error).toExponential(1)}`;
  $('projection-note').textContent=`Anchor projection (log): ${Number(result.diagnostics.anchor_log_projection).toExponential(2)}; shape adjustment: ${fmt(result.diagnostics.projection_change,4)}.`;
  await draw();exportEnabled(true);
}
async function draw() {
  const items=state.current?[state.current,...state.comparisons.filter(r=>JSON.stringify(r.inputs)!==JSON.stringify(state.current.inputs))]:[];
  const traces=items.map((r,i)=>({x:r.curve.time_ms.map(t=>state.normalized?t/r.anchors.Td:t),y:r.curve.force_kN.map(f=>state.normalized?f/r.anchors.Fmax:f),type:'scatter',mode:'lines',name:r.name,line:{color:colors[i],width:i===0?2.6:1.8,dash:i===0?'solid':'dash'},hovertemplate:state.normalized?'t/Td = %{x:.4f}<br>F/Fmax = %{y:.4f}<extra>%{fullData.name}</extra>':'Time = %{x:.3f} ms<br>Force = %{y:.3f} kN<extra>%{fullData.name}</extra>'}));
  const layout={autosize:true,paper_bgcolor:'white',plot_bgcolor:'white',font:{family:'Arial, Microsoft YaHei, sans-serif',size:12,color:'#475353'},margin:{l:65,r:24,t:22,b:60},showlegend:items.length>1,legend:{orientation:'h',x:0,y:1.12},hovermode:'closest',xaxis:{title:{text:state.normalized?'Normalized time, t / T<sub>d</sub>':'Time, t (ms)',standoff:14},rangemode:'tozero',gridcolor:'#edf0f0',zeroline:false,showline:true,linecolor:'#a8b5b4',ticks:'outside',tickcolor:'#a8b5b4',automargin:true},yaxis:{title:{text:state.normalized?'Normalized force, F / F<sub>max</sub>':'Impact force, F (kN)',standoff:12},rangemode:'tozero',gridcolor:'#edf0f0',zeroline:false,showline:true,linecolor:'#a8b5b4',ticks:'outside',tickcolor:'#a8b5b4',automargin:true}};
  await Plotly.react($('curve-chart'),traces,layout,{responsive:true,displayModeBar:false,scrollZoom:false,displaylogo:false});
}
function renderComparisons() {
  $('comparison-list').replaceChildren();$('comparison-list').hidden=state.comparisons.length===0;
  state.comparisons.forEach((r,i)=>{const chip=document.createElement('span');chip.className='compare-chip';const dot=document.createElement('span');dot.className='compare-dot';dot.style.background=colors[i+1];const name=document.createElement('span');name.textContent=r.name;const remove=document.createElement('button');remove.title='Remove comparison curve';remove.setAttribute('aria-label','Remove '+r.name);remove.innerHTML='<i data-lucide="x"></i>';remove.addEventListener('click',()=>{state.comparisons.splice(i,1);renderComparisons();draw();});chip.append(dot,name,remove);$('comparison-list').append(chip);});icons();
}
function cleaned(result) {const {curve_csv,elapsed_ms,...rest}=result;return rest;}
function modelInfo() {
  $('performance-table').replaceChildren();
  const rows=specs.map(([key,label])=>[key,label]);rows.push(['median_curve_R2','Median curve R² (physical time)']);
  rows.forEach(([key,label])=>{const tr=document.createElement('tr');for(const value of [label,fmt(state.meta.performance.validation[key]),fmt(state.meta.performance.test[key])]){const td=document.createElement('td');td.textContent=value;tr.append(td);}$('performance-table').append(tr);});
  $('model-hash').textContent=state.meta.model_sha256;$('info-dialog').showModal();
}
async function batchFile(file) {
  $('batch-error').hidden=true;$('batch-select').disabled=true;$('batch-select').lastChild.textContent='Computing';
  state.batch=null;$('batch-results').hidden=true;$('batch-empty').hidden=false;$('batch-count').textContent='0 cases';$('batch-table').replaceChildren();
  try {
    const result=await request('/api/batch',await file.text(),true);state.batch=result;$('batch-empty').hidden=true;$('batch-results').hidden=false;$('batch-count').textContent=result.results.length+' cases';$('batch-table').replaceChildren();
    for(const r of result.results){const tr=document.createElement('tr');const name=document.createElement('td');name.textContent=r.name;tr.append(name);specs.forEach(([k,,,,d])=>{const td=document.createElement('td');td.textContent=fmt(r.anchors[k],d);tr.append(td);});const status=document.createElement('td');status.textContent=r.warnings.length?`${r.warnings.length} notices`:'Complete';const action=document.createElement('td');const button=document.createElement('button');button.className='text-button';button.textContent='View curve';button.addEventListener('click',()=>{fill(r.inputs,r.name);switchTab('single');present(r);});action.append(button);tr.append(status,action);$('batch-table').append(tr);}
    toast('Batch prediction complete');
  }catch(e){showError(e,'batch-error');}
  finally{$('batch-select').disabled=false;$('batch-select').lastChild.textContent='Import & predict';$('batch-file').value='';}
}
function bind() {
  $('parameter-form').addEventListener('submit',e=>{e.preventDefault();predict();});
  $('case-name').addEventListener('input',stale);
  document.querySelectorAll('[data-tab]').forEach(b=>b.addEventListener('click',()=>switchTab(b.dataset.tab)));
  $('example-select').addEventListener('change',()=>{const e=state.meta.examples[Number($('example-select').value)];if($('example-select').value!==''&&e) {fill(e.inputs,e.name);predict();}});
  $('reset-form').addEventListener('click',()=>{fill({},'New case');$('example-select').value='';});
  $('import-params').addEventListener('click',()=>$('params-file').click());
  $('params-file').addEventListener('change',async e=>{const file=e.target.files[0];if(!file)return;try{const data=JSON.parse(await file.text());const values=data.inputs||data;const unknown=Object.keys(values).filter(k=>!state.meta.fields.some(f=>f.key===k));if(unknown.length)throw new Error('Unknown parameters: '+unknown.join(', '));fill(values,data.name||file.name.replace(/\.json$/i,''));$('example-select').value='';toast('Parameters imported');}catch(error){showError(error);}e.target.value='';});
  $('save-params').addEventListener('click',()=>download(JSON.stringify({name:$('case-name').value,inputs:readInputs()},null,2),safeName($('case-name').value)+'_inputs.json','application/json;charset=utf-8'));
  $('physical-mode').addEventListener('click',()=>{state.normalized=false;$('physical-mode').classList.add('active');$('normalized-mode').classList.remove('active');draw();});
  $('normalized-mode').addEventListener('click',()=>{state.normalized=true;$('normalized-mode').classList.add('active');$('physical-mode').classList.remove('active');draw();});
  $('reset-zoom').addEventListener('click',()=>Plotly.relayout($('curve-chart'),{'xaxis.autorange':true,'yaxis.autorange':true}));
  $('export-csv').addEventListener('click',()=>download(state.current.curve_csv,safeName(state.current.name)+'_force_history.csv'));
  $('export-json').addEventListener('click',()=>download(JSON.stringify(cleaned(state.current),null,2),safeName(state.current.name)+'_prediction.json','application/json;charset=utf-8'));
  for(const type of ['png','svg']) $('export-'+type).addEventListener('click',()=>Plotly.downloadImage($('curve-chart'),{format:type,width:1500,height:850,filename:safeName(state.current.name)+(state.normalized?'_normalized':'_force_history')}));
  $('compare-add').addEventListener('click',()=>{if(state.comparisons.length>=5)return toast('Up to 5 reference curves can be retained');if(state.comparisons.some(r=>JSON.stringify(r.inputs)===JSON.stringify(state.current.inputs)))return toast('This case is already in the comparison');state.comparisons.push(structuredClone(state.current));renderComparisons();draw();toast('Curve added to comparison');});
  $('model-info').addEventListener('click',modelInfo);$('close-info').addEventListener('click',()=>$('info-dialog').close());
  $('shutdown').addEventListener('click',async()=>{if(!confirm('Stop the local prediction service? Exported files will not be affected.'))return;try{await request('/api/shutdown',{});toast('Local service stopped');document.querySelectorAll('button').forEach(b=>b.disabled=true);$('prediction-state').textContent='Service stopped';}catch(e){toast(e.message);}});
  $('batch-template').addEventListener('click',async()=>{const response=await fetch('/api/template');download((await response.text()).replace(/^\ufeff/,''),'ImpactCurve_batch_template.csv');});
  $('batch-select').addEventListener('click',()=>$('batch-file').click());$('batch-file').addEventListener('change',e=>{if(e.target.files[0])batchFile(e.target.files[0]);});
  $('batch-summary').addEventListener('click',()=>download(state.batch.summary_csv,'ImpactCurve_batch_anchors.csv'));
  $('batch-curves').addEventListener('click',()=>download(state.batch.curves_csv,'ImpactCurve_batch_curves.csv'));
}
async function init() {
  try{state.meta=await request('/api/meta');fields();bind();state.meta.examples.forEach((e,i)=>{const option=document.createElement('option');option.value=i;option.textContent=e.name;$('example-select').append(option);});icons();await draw();let lastWidth=0;new ResizeObserver(entries=>{const width=entries[0].contentRect.width;if(width>0&&Math.abs(width-lastWidth)>1){lastWidth=width;Plotly.Plots.resize($('curve-chart'));}}).observe(document.querySelector('.chart-wrap'));fill(state.meta.examples[0].inputs,state.meta.examples[0].name);$('example-select').value='0';await predict();}
  catch(error){$('prediction-state').textContent='Service unavailable';$('input-status').textContent='Not ready';showError(error);}
}
init();
