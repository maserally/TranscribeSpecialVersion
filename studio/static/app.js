const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
let sourceMode='path', uploadedPath='';
let localWhisperModels=[], installedWhisperModels=[], localOllamaModels=[];
let lastAsrKind='local_whisper';
let lastTranslatorKind='local_ollama';
const providerUrls={local_ollama:'http://127.0.0.1:11434',openai_compatible:'https://api.openai.com/v1'};
const outputLabels={soft_video:'软字幕视频',hard_video:'硬字幕视频',publish_cn_srt:'观看版中文字幕',publish_bilingual_srt:'观看版双语字幕',publish_ja_srt:'观看版日文字幕',review_cn_srt:'校对版中文字幕',review_bilingual_srt:'校对版双语字幕',review_ja_srt:'校对版日文字幕',quality_report:'质量报告'};
const statusLabels={queued:'等待处理',running:'处理中',completed:'已完成',failed:'失败'};
const whisperHelp={tiny:'速度最快、资源占用最低，日语准确率较低',base:'速度很快，适合清晰短语音',small:'较快，适合初步识别；本机已安装时可离线使用',medium:'默认推荐，日语准确率和速度较均衡',large:'旧版大型模型，通常建议改用 large-v3', 'large-v2':'上一代高精度模型', 'large-v3':'当前本地高精度复核首选，显存占用较高',turbo:'大型模型的加速版本，速度快但时间戳表现因素材而异'};

async function jsonFetch(url,options={}){const r=await fetch(url,options);const data=await r.json().catch(()=>({}));if(!r.ok)throw new Error(data.detail||`HTTP ${r.status}`);return data}
function fillModelSelect(id,models,installed=null,preferred='',allowCustom=true){const el=$(id),local=new Set(installed||[]),current=modelValue(id),showLocalState=Array.isArray(installed);el.innerHTML=models.map(x=>`<option value="${esc(x)}">${esc(x)}${showLocalState?(local.has(x)?'（本机已安装）':'（首次使用可能下载）'):''}</option>`).join('')+(allowCustom?'<option value="__custom__">手动输入模型 ID…</option>':'');const next=models.includes(current)?current:(models.includes(preferred)?preferred:(models[0]||'__custom__'));el.value=next;syncCustomModel(id)}
function modelValue(id){const el=$(id);if(!el)return'';if(el.value==='__custom__'){const custom=$(id.replace('-model','-model-custom'));return custom?custom.value.trim():''}return el.value}
function syncCustomModel(id){const custom=$(id.replace('-model','-model-custom'));if(custom)custom.classList.toggle('hidden',$(id).value!=='__custom__')}

async function init(){
  const health=await jsonFetch('/api/health'); $('#health').textContent=health.ok?`环境正常 · ${health.gpu||'CPU'}`:'缺少 FFmpeg';
  const local=await jsonFetch('/api/models/local');localWhisperModels=local.whisper;installedWhisperModels=local.whisper_installed;localOllamaModels=local.ollama;fillModelSelect('#asr-model',localWhisperModels,installedWhisperModels,'medium');fillModelSelect('#verifier-model',localWhisperModels,installedWhisperModels,'large-v3',false);fillModelSelect('#translator-model',localOllamaModels,localOllamaModels,'qwen2.5:7b-instruct');
  $('#installed-whisper').textContent=local.whisper_installed.length?`本机已安装：${local.whisper_installed.join('、')}　其他列表项首次使用时需要下载`:'本机尚未缓存 Whisper 模型；首次运行所选模型时需要下载';
  updateModelHelp();
  renderJobs(); setInterval(renderJobs,2500);
}

$$('.tab').forEach(btn=>btn.onclick=()=>{sourceMode=btn.dataset.mode;$$('.tab').forEach(x=>x.classList.toggle('active',x===btn));$$('.source-view').forEach(x=>x.classList.remove('active'));$(`#${sourceMode}-source`).classList.add('active')});
$$('input[name=profile]').forEach(x=>x.onchange=()=>$$('.profile').forEach(p=>p.classList.toggle('selected',p.contains(x)&&x.checked)));

function syncProviderFields(){
  const ak=$('#asr-kind').value, tk=$('#translator-kind').value;
  $$('[data-for=asr]').forEach(x=>x.classList.toggle('hidden',ak!=='openai_compatible'));
  $$('[data-for=translator]').forEach(x=>x.classList.toggle('hidden',tk!=='openai_compatible'));
  if(ak!==lastAsrKind){
    if(ak==='local_whisper')fillModelSelect('#asr-model',localWhisperModels,installedWhisperModels,'medium');
    else fillModelSelect('#asr-model',[],null,'');
    lastAsrKind=ak;
  }
  if(tk!==lastTranslatorKind){
    providerUrls[lastTranslatorKind]=$('#translator-url').value.trim()||providerUrls[lastTranslatorKind];
    $('#translator-url').value=providerUrls[tk]||(tk==='local_ollama'?'http://127.0.0.1:11434':'https://api.openai.com/v1');
    if(tk==='local_ollama')fillModelSelect('#translator-model',localOllamaModels,localOllamaModels,'qwen2.5:7b-instruct');
    else fillModelSelect('#translator-model',[],null,'');
    lastTranslatorKind=tk;
  }
  updateModelHelp();
}
$('#asr-kind').onchange=syncProviderFields; $('#translator-kind').onchange=syncProviderFields;

async function refreshModels(kind){
  const isAsr=kind==='asr', provider={kind:isAsr?$('#asr-kind').value:$('#translator-kind').value,base_url:isAsr?$('#asr-url').value:$('#translator-url').value,api_key:isAsr?$('#asr-key').value:$('#translator-key').value,model:''};
  try{const data=await jsonFetch('/api/models',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({provider})});const localState=provider.kind==='local_whisper'?installedWhisperModels:(provider.kind==='local_ollama'?data.models:null);fillModelSelect(isAsr?'#asr-model':'#translator-model',data.models,localState,data.models[0]||'');updateModelHelp()}catch(e){$('#form-error').textContent=`模型列表读取失败：${e.message}`}
}
$('#refresh-asr').onclick=()=>refreshModels('asr'); $('#refresh-translator').onclick=()=>refreshModels('translator');

function updateModelHelp(){
  const asr=modelValue('#asr-model'), verifier=modelValue('#verifier-model');
  $('#asr-model-help').textContent=$('#asr-kind').value==='local_whisper'?(whisperHelp[asr]||'本地 Whisper 模型；未缓存时首次使用需要下载'):'远程语音识别模型；接口必须支持分段时间戳';
  $('#verifier-model-help').textContent=`${whisperHelp[verifier]||'本地 Whisper 复核模型'}；只复核初筛对白，不会全片重复识别`;
  $('#translator-model-help').textContent=$('#translator-kind').value==='local_ollama'?'本地逐句日译中；请先在 Ollama 中安装所选模型':'远程逐句日译中；需要支持 Chat Completions 和 JSON 输出';
}
['#asr-model','#verifier-model','#translator-model'].forEach(id=>$(id).addEventListener('change',()=>{syncCustomModel(id);updateModelHelp()}));['#asr-model-custom','#translator-model-custom'].forEach(id=>$(id).addEventListener('input',updateModelHelp));

async function uploadIfNeeded(){
  if(sourceMode==='path')return $('#input-path').value.trim(); if(uploadedPath)return uploadedPath;
  const file=$('#upload-file').files[0]; if(!file)throw new Error('请选择视频'); const form=new FormData();form.append('file',file);
  return new Promise((resolve,reject)=>{const xhr=new XMLHttpRequest();xhr.open('POST','/api/uploads');xhr.upload.onprogress=e=>{if(e.lengthComputable)$('#upload-progress').style.width=`${e.loaded/e.total*100}%`;};xhr.onload=()=>{if(xhr.status<300){const d=JSON.parse(xhr.responseText);uploadedPath=d.path;$('#upload-status').textContent=`已上传 ${(d.size/1024/1024).toFixed(1)} MB`;resolve(d.path)}else reject(new Error('上传失败'))};xhr.onerror=()=>reject(new Error('上传失败'));xhr.send(form)});
}

$('#start').onclick=async()=>{
  $('#form-error').textContent=''; $('#start').disabled=true;
  try{const input_path=await uploadIfNeeded();const asrKind=$('#asr-kind').value, transKind=$('#translator-kind').value;const asrModel=modelValue('#asr-model'),translatorModel=modelValue('#translator-model');if(!asrModel||!translatorModel)throw new Error('请选择或输入识别模型和翻译模型');const body={input_path,output_name:$('#output-name').value.trim(),source_language:'ja',target_language:'zh-CN',profile:$('input[name=profile]:checked').value,asr:{kind:asrKind,base_url:asrKind==='openai_compatible'?$('#asr-url').value:'',api_key:$('#asr-key').value,model:asrModel},verifier_model:modelValue('#verifier-model'),translator:{kind:transKind,base_url:$('#translator-url').value,api_key:$('#translator-key').value,model:translatorModel},remove_chinese_periods:$('#remove-periods').checked,publish_mode:$('#publish-mode').checked,create_soft_subtitle_video:$('#soft-video').checked,create_hard_subtitle_video:$('#hard-video').checked,enable_gap_recovery:$('#gap-recovery').checked};await jsonFetch('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});await renderJobs()}catch(e){$('#form-error').textContent=e.message}finally{$('#start').disabled=false}
};

function esc(s){return String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
async function archiveJob(id){if(!confirm('隐藏这条任务记录？产物会移到 studio_data/archive，可手动恢复。'))return;try{await jsonFetch(`/api/jobs/${id}`,{method:'DELETE'});await renderJobs()}catch(e){alert(e.message)}}
async function renderJobs(){
  const data=await jsonFetch('/api/jobs').catch(()=>({jobs:[]}));const box=$('#jobs-list');if(!data.jobs.length){box.innerHTML='<p class="muted">尚无任务</p>';return}
  box.innerHTML=data.jobs.map(j=>`<article class="job"><div class="job-top"><div><b>${esc(j.options.output_name||j.options.input_path.split(/[\\/]/).pop())}</b><div class="muted">${esc(j.stage)}</div></div><span class="badge">${esc(statusLabels[j.status]||j.status)}</span></div><div class="progress"><i style="width:${j.progress*100}%"></i></div>${j.error?`<details><summary class="error">历史错误详情（不会影响新任务）</summary><pre class="logs">${esc(j.error)}</pre></details>`:''}<details><summary>运行日志</summary><pre class="logs">${esc(j.logs.join('\n'))}</pre></details><div class="downloads">${Object.keys(j.outputs).filter(k=>outputLabels[k]).map(k=>`<a href="/api/jobs/${j.id}/download/${k}">${outputLabels[k]}</a>`).join('')}${['completed','failed'].includes(j.status)?`<button class="remove" onclick="archiveJob('${j.id}')">隐藏记录</button>`:''}</div></article>`).join('')
}
$('#refresh-jobs').onclick=renderJobs; init().catch(e=>$('#health').textContent=e.message);
