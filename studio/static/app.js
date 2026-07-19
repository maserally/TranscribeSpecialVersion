const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
let sourceMode='path', uploadedPath='';
let runtimeInfo={mode:'local',local_open:true,local_path_input:true,secret_policy:'windows_dpapi'};
let localWhisperModels=[], installedWhisperModels=[], localOllamaModels=[];
let lastAsrKind='local_whisper';
let lastTranslatorKind='local_ollama';
let lastTextReviewerKind='local_ollama';
const openJobDetails=new Set();
const providerUrls={local_ollama:'http://127.0.0.1:11434',openai_compatible:'https://api.openai.com/v1'};
const reviewerUrls={local_ollama:'http://127.0.0.1:11434',openai_compatible:'https://api.openai.com/v1'};
const languageMeta={ja:{name:'日语',pair:'日译中'},ko:{name:'韩语',pair:'韩译中'}};
const outputLabels={soft_video:'软字幕视频',hard_video:'硬字幕视频',publish_cn_srt:'观看版中文字幕',publish_bilingual_srt:'观看版双语字幕',publish_source_srt:'观看版原文字幕',publish_ja_srt:'观看版日文字幕',review_cn_srt:'校对版中文字幕',review_bilingual_srt:'校对版双语字幕',review_source_srt:'校对版原文字幕',review_ja_srt:'校对版日文字幕',publish_json:'观看版字幕数据',review_json:'校对版字幕数据',quality_report:'质量报告',text_review_audit:'最终文本校正记录'};
const statusLabels={queued:'等待处理',running:'处理中',paused:'已暂停',canceled:'已取消',completed:'已完成',failed:'失败'};
const whisperHelp={tiny:'速度最快、资源占用最低，准确率较低',base:'速度很快，适合清晰短语音',small:'较快，适合初步识别；本机已安装时可离线使用',medium:'默认推荐，准确率和速度较均衡',large:'旧版大型模型，通常建议改用 large-v3', 'large-v2':'上一代高精度模型', 'large-v3':'当前本地高精度复核首选，显存占用较高',turbo:'大型模型的加速版本，速度快但时间戳表现因素材而异'};

async function jsonFetch(url,options={}){const r=await fetch(url,options);const data=await r.json().catch(()=>({}));if(!r.ok)throw new Error(data.detail||`HTTP ${r.status}`);return data}
function fillModelSelect(id,models,installed=null,preferred='',allowCustom=true){const el=$(id),local=new Set(installed||[]),current=modelValue(id),showLocalState=Array.isArray(installed);el.innerHTML=models.map(x=>`<option value="${esc(x)}">${esc(x)}${showLocalState?(local.has(x)?'（本机已安装）':'（首次使用可能下载）'):''}</option>`).join('')+(allowCustom?'<option value="__custom__">手动输入模型 ID…</option>':'');const next=models.includes(current)?current:(models.includes(preferred)?preferred:(models[0]||'__custom__'));el.value=next;syncCustomModel(id)}
function modelValue(id){const el=$(id);if(!el)return'';if(el.value==='__custom__'){const custom=$(id.replace('-model','-model-custom'));return custom?custom.value.trim():''}return el.value}
function syncCustomModel(id){const custom=$(id.replace('-model','-model-custom'));if(custom)custom.classList.toggle('hidden',$(id).value!=='__custom__')}
function fillSavedModel(id,models,installed,value,allowCustom=true){fillModelSelect(id,models,installed,value,allowCustom);if(value&&!models.includes(value)&&allowCustom){$(id).value='__custom__';const custom=$(id.replace('-model','-model-custom'));if(custom)custom.value=value;syncCustomModel(id)}}

function providerSettingsBody(){return{asr:{kind:$('#asr-kind').value,base_url:$('#asr-url').value.trim(),api_key:$('#asr-key').value,model:modelValue('#asr-model')},translator:{kind:$('#translator-kind').value,base_url:$('#translator-url').value.trim(),api_key:$('#translator-key').value,model:modelValue('#translator-model')},text_reviewer:{kind:$('#text-reviewer-kind').value,base_url:$('#text-reviewer-url').value.trim(),api_key:$('#text-reviewer-key').value,model:modelValue('#text-reviewer-model')},verifier_model:modelValue('#verifier-model')}}
async function saveProviderSettings(showStatus=true){const data=await jsonFetch('/api/settings/providers',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(providerSettingsBody())});if(showStatus)$('#provider-save-status').textContent=data.secret_policy==='environment'?'模型与地址已保存；云端 API Key 请通过环境变量配置':`已保存到 ${data.path}`;return data}
const saveProviderButton=$('#save-provider-settings');if(saveProviderButton)saveProviderButton.onclick=async()=>{try{await saveProviderSettings(true)}catch(e){const status=$('#provider-save-status');if(status)status.textContent=`保存失败：${e.message}`}};

function cleanInputPath(value){let path=String(value||'').trim();const pairs=[['"','"'],["'","'"],['“','”'],['‘','’']];let changed=true;while(changed&&path.length>1){changed=false;for(const [left,right] of pairs){if(path.startsWith(left)&&path.endsWith(right)){path=path.slice(left.length,-right.length).trim();changed=true}}}if(/^file:\/\//i.test(path)){try{path=decodeURIComponent(new URL(path).pathname).replace(/^\/(?:([A-Za-z]:))/,'$1')}catch{}}if(/^[A-Za-z]:\//.test(path))path=path.replaceAll('/','\\');return path}
function cleanPathField(){const input=$('#input-path'),before=input.value,after=cleanInputPath(before);input.value=after;const status=$('#path-clean-status');if(status)status.textContent=before===after?'路径格式无需修改':'已移除外层引号或多余空格';return after}
const cleanPathButton=$('#clean-input-path');if(cleanPathButton)cleanPathButton.onclick=cleanPathField;const inputPath=$('#input-path');if(inputPath)inputPath.addEventListener('blur',cleanPathField);

async function init(){
  runtimeInfo=await jsonFetch('/api/runtime');
  if(runtimeInfo.mode==='cloud'){
    sourceMode='upload';
    $('#cloud-notice').classList.remove('hidden');
    const pathTab=$('.tab[data-mode=path]');if(pathTab)pathTab.classList.add('hidden');
    $$('.tab').forEach(x=>x.classList.toggle('active',x.dataset.mode==='upload'));
    $$('.source-view').forEach(x=>x.classList.toggle('active',x.id==='upload-source'));
    $('#soft-video').checked=false;$('#hard-video').checked=false;
    $('#provider-storage-title').textContent='6. 云端接口配置';
    $('#provider-storage-help').textContent='模型与 Base URL 保存到云端数据目录；API Key 不写入磁盘，请使用 SUBTITLE_ASR_API_KEY、SUBTITLE_TRANSLATOR_API_KEY 和 SUBTITLE_REVIEWER_API_KEY 环境变量。';
  }
  const health=await jsonFetch('/api/health'); $('#health').textContent=health.ok?`环境正常 · ${health.gpu||'CPU'}`:'缺少 FFmpeg';
  const local=await jsonFetch('/api/models/local');localWhisperModels=local.whisper;installedWhisperModels=local.whisper_installed;localOllamaModels=local.ollama;
  const saved=await jsonFetch('/api/settings/providers').catch(()=>null);
  const asr=saved?.asr||{kind:'local_whisper',base_url:'https://api.openai.com/v1',api_key:'',model:'medium'},translator=saved?.translator||{kind:'local_ollama',base_url:'http://127.0.0.1:11434',api_key:'',model:'qwen2.5:7b-instruct'},textReviewer=saved?.text_reviewer||{kind:'local_ollama',base_url:'http://127.0.0.1:11434',api_key:'',model:'qwen2.5:7b-instruct'};
  providerUrls[translator.kind]=translator.base_url||providerUrls[translator.kind];
  reviewerUrls[textReviewer.kind]=textReviewer.base_url||reviewerUrls[textReviewer.kind];
  $('#asr-kind').value=asr.kind;$('#asr-url').value=asr.base_url;$('#asr-key').value=asr.api_key||'';$('#translator-kind').value=translator.kind;$('#translator-url').value=translator.base_url;$('#translator-key').value=translator.api_key||'';$('#text-reviewer-kind').value=textReviewer.kind;$('#text-reviewer-url').value=textReviewer.base_url;$('#text-reviewer-key').value=textReviewer.api_key||'';lastAsrKind=asr.kind;lastTranslatorKind=translator.kind;lastTextReviewerKind=textReviewer.kind;
  if(asr.api_key_configured)$('#asr-key').placeholder='已通过云端环境变量配置';if(translator.api_key_configured)$('#translator-key').placeholder='已通过云端环境变量配置';if(textReviewer.api_key_configured)$('#text-reviewer-key').placeholder='已通过云端环境变量配置';
  fillSavedModel('#asr-model',asr.kind==='local_whisper'?localWhisperModels:(asr.model?[asr.model]:[]),asr.kind==='local_whisper'?installedWhisperModels:null,asr.model||'medium');
  fillSavedModel('#verifier-model',localWhisperModels,installedWhisperModels,saved?.verifier_model||'large-v3',false);
  fillSavedModel('#translator-model',translator.kind==='local_ollama'?localOllamaModels:(translator.model?[translator.model]:[]),translator.kind==='local_ollama'?localOllamaModels:null,translator.model||'qwen2.5:7b-instruct');
  fillSavedModel('#text-reviewer-model',textReviewer.kind==='local_ollama'?localOllamaModels:(textReviewer.model?[textReviewer.model]:[]),textReviewer.kind==='local_ollama'?localOllamaModels:null,textReviewer.model||'qwen2.5:7b-instruct');
  syncProviderFields();
  $('#installed-whisper').textContent=local.whisper_installed.length?`本机已安装：${local.whisper_installed.join('、')}　其他列表项首次使用时需要下载`:'本机尚未缓存 Whisper 模型；首次运行所选模型时需要下载';
  updateModelHelp();
  renderJobs(); setInterval(renderJobs,2500);
}

$$('.tab').forEach(btn=>btn.onclick=()=>{sourceMode=btn.dataset.mode;$$('.tab').forEach(x=>x.classList.toggle('active',x===btn));$$('.source-view').forEach(x=>x.classList.remove('active'));$(`#${sourceMode}-source`).classList.add('active')});
$$('input[name=profile]').forEach(x=>x.onchange=()=>$$('.profile').forEach(p=>p.classList.toggle('selected',p.contains(x)&&x.checked)));

function syncProviderFields(){
  const ak=$('#asr-kind').value, tk=$('#translator-kind').value, rk=$('#text-reviewer-kind').value;
  $$('[data-for=asr]').forEach(x=>x.classList.toggle('hidden',ak!=='openai_compatible'));
  $$('[data-for=translator]').forEach(x=>x.classList.toggle('hidden',tk!=='openai_compatible'));
  $$('[data-for=text-reviewer]').forEach(x=>x.classList.toggle('hidden',rk!=='openai_compatible'));
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
  if(rk!==lastTextReviewerKind){
    reviewerUrls[lastTextReviewerKind]=$('#text-reviewer-url').value.trim()||reviewerUrls[lastTextReviewerKind];
    $('#text-reviewer-url').value=reviewerUrls[rk]||(rk==='local_ollama'?'http://127.0.0.1:11434':'https://api.openai.com/v1');
    if(rk==='local_ollama')fillModelSelect('#text-reviewer-model',localOllamaModels,localOllamaModels,'qwen2.5:7b-instruct');
    else fillModelSelect('#text-reviewer-model',[],null,'');
    lastTextReviewerKind=rk;
  }
  updateModelHelp();
}
$('#asr-kind').onchange=syncProviderFields; $('#translator-kind').onchange=syncProviderFields; $('#text-reviewer-kind').onchange=syncProviderFields;

async function refreshModels(kind){
  const prefix=kind==='asr'?'asr':(kind==='translator'?'translator':'text-reviewer'),provider={kind:$(`#${prefix}-kind`).value,base_url:$(`#${prefix}-url`).value,api_key:$(`#${prefix}-key`).value,model:''};
  try{const role=kind==='text-reviewer'?'text_reviewer':kind;const data=await jsonFetch('/api/models',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({provider,role})});const localState=provider.kind==='local_whisper'?installedWhisperModels:(provider.kind==='local_ollama'?data.models:null);fillModelSelect(`#${prefix}-model`,data.models,localState,data.models[0]||'');updateModelHelp()}catch(e){$('#form-error').textContent=`模型列表读取失败：${e.message}`}
}
$('#refresh-asr').onclick=()=>refreshModels('asr'); $('#refresh-translator').onclick=()=>refreshModels('translator'); $('#refresh-text-reviewer').onclick=()=>refreshModels('text-reviewer');

function updateModelHelp(){
  const language=languageMeta[$('#source-language').value]||languageMeta.ja;
  const asr=modelValue('#asr-model'), verifier=modelValue('#verifier-model');
  $('#asr-title').textContent=`3. ${language.name}识别模型`;
  $('#asr-model-help').textContent=$('#asr-kind').value==='local_whisper'?(whisperHelp[asr]||'本地 Whisper 模型；未缓存时首次使用需要下载'):(asr.startsWith('gpt-4o')&&asr.includes('transcribe')?'gpt-4o-transcribe 使用逐个 VAD 窗口转写，时间轴精度略低于 Whisper 分段时间戳':'远程模型如支持 verbose_json.segments 将使用精细时间轴');
  $('#verifier-model-help').textContent=`${whisperHelp[verifier]||'本地 Whisper 复核模型'}；只复核初筛对白，不会全片重复识别`;
  $('#translator-model-help').textContent=$('#translator-kind').value==='local_ollama'?`本地逐句${language.pair}；请先在 Ollama 中安装所选模型`:`远程逐句${language.pair}；需要支持 Chat Completions 和 JSON 输出`;
  $('#text-reviewer-model-help').textContent=$('#text-reviewer-kind').value==='local_ollama'?'自动本地校正；按批次读取前后文，只改文字和术语一致性':'自动云端校正；需要支持 Chat Completions 和 JSON 输出，不修改时间轴';
}
$('#source-language').onchange=updateModelHelp;
['#asr-model','#verifier-model','#translator-model','#text-reviewer-model'].forEach(id=>$(id).addEventListener('change',()=>{syncCustomModel(id);updateModelHelp()}));['#asr-model-custom','#translator-model-custom','#text-reviewer-model-custom'].forEach(id=>$(id).addEventListener('input',updateModelHelp));

async function uploadIfNeeded(){
  if(sourceMode==='path')return cleanPathField(); if(uploadedPath)return uploadedPath;
  const file=$('#upload-file').files[0]; if(!file)throw new Error('请选择视频'); const form=new FormData();form.append('file',file);
  return new Promise((resolve,reject)=>{const xhr=new XMLHttpRequest();xhr.open('POST','/api/uploads');xhr.upload.onprogress=e=>{if(e.lengthComputable)$('#upload-progress').style.width=`${e.loaded/e.total*100}%`;};xhr.onload=()=>{if(xhr.status<300){const d=JSON.parse(xhr.responseText);uploadedPath=d.path;$('#upload-status').textContent=`已上传 ${(d.size/1024/1024).toFixed(1)} MB`;resolve(d.path)}else reject(new Error('上传失败'))};xhr.onerror=()=>reject(new Error('上传失败'));xhr.send(form)});
}

$('#start').onclick=async()=>{
  $('#form-error').textContent=''; $('#start').disabled=true;
  try{const input_path=await uploadIfNeeded();const asrKind=$('#asr-kind').value, transKind=$('#translator-kind').value,reviewKind=$('#text-reviewer-kind').value;const asrModel=modelValue('#asr-model'),translatorModel=modelValue('#translator-model'),reviewModel=modelValue('#text-reviewer-model');if(!asrModel||!translatorModel||!reviewModel)throw new Error('请选择或输入识别、翻译和自动校正模型');await saveProviderSettings(false);const body={input_path,output_name:$('#output-name').value.trim(),source_language:$('#source-language').value,target_language:'zh-CN',profile:$('input[name=profile]:checked').value,asr:{kind:asrKind,base_url:asrKind==='openai_compatible'?$('#asr-url').value:'',api_key:$('#asr-key').value,model:asrModel},verifier_model:modelValue('#verifier-model'),translator:{kind:transKind,base_url:$('#translator-url').value,api_key:$('#translator-key').value,model:translatorModel},text_reviewer:{kind:reviewKind,base_url:$('#text-reviewer-url').value,api_key:$('#text-reviewer-key').value,model:reviewModel},remove_chinese_periods:$('#remove-periods').checked,publish_mode:$('#publish-mode').checked,create_soft_subtitle_video:$('#soft-video').checked,create_hard_subtitle_video:$('#hard-video').checked};await jsonFetch('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});await renderJobs()}catch(e){$('#form-error').textContent=e.message}finally{$('#start').disabled=false}
};

function esc(s){return String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
async function jobAction(id,action){try{await jsonFetch(`/api/jobs/${id}/${action}`,{method:'POST'});await renderJobs()}catch(e){alert(e.message)}}
async function pauseJob(id){await jobAction(id,'pause')}
async function resumeJob(id){await jobAction(id,'resume')}
async function cancelJob(id){if(confirm('确定取消这个任务？当前阶段会停止，已生成的中间文件会保留到你删除任务为止。'))await jobAction(id,'cancel')}
async function deleteJob(id){if(!confirm('永久删除这条任务及其本地中间文件和产物？若视频由网页上传，未被其他任务使用的上传副本也会删除。此操作不可恢复。'))return;try{await jsonFetch(`/api/jobs/${id}`,{method:'DELETE'});[...openJobDetails].filter(x=>x.startsWith(`${id}:`)).forEach(x=>openJobDetails.delete(x));await renderJobs()}catch(e){alert(e.message)}}
async function openOutput(id,key){try{await jsonFetch(`/api/jobs/${id}/open/${encodeURIComponent(key)}`,{method:'POST'})}catch(e){alert(`打开失败：${e.message}`)}}
async function openOutputFolder(id){try{await jsonFetch(`/api/jobs/${id}/open-folder`,{method:'POST'})}catch(e){alert(`打开文件夹失败：${e.message}`)}}
function detailMarkup(id,type,title,content,titleClass=''){const key=`${id}:${type}`;return`<details data-open-key="${key}" ${openJobDetails.has(key)?'open':''}><summary class="${titleClass}">${title}</summary><pre class="logs">${esc(content)}</pre></details>`}
function actionMarkup(j){if(['queued','running'].includes(j.status))return`<button onclick="pauseJob('${j.id}')">暂停</button><button class="danger" onclick="cancelJob('${j.id}')">取消</button>`;if(j.status==='paused')return`<button onclick="resumeJob('${j.id}')">继续</button><button class="danger" onclick="cancelJob('${j.id}')">取消</button>`;return`<button class="danger" onclick="deleteJob('${j.id}')">删除任务</button>`}
function outputMarkup(id,key){return runtimeInfo.local_open?`<button onclick="openOutput('${id}','${key}')">打开${outputLabels[key]}</button>`:`<a href="/api/jobs/${id}/download/${encodeURIComponent(key)}">下载${outputLabels[key]}</a>`}
async function renderJobs(){
  const data=await jsonFetch('/api/jobs').catch(()=>({jobs:[]}));const box=$('#jobs-list');if(!data.jobs.length){box.innerHTML='<p class="muted">尚无任务</p>';return}
  box.innerHTML=data.jobs.map(j=>`<article class="job"><div class="job-top"><div><b>${esc(j.options.output_name||j.options.input_path.split(/[\\/]/).pop())}</b><div class="muted">${esc(languageMeta[j.options.source_language]?.name||'日语')} · ${esc(j.stage)}</div></div><span class="badge">${esc(statusLabels[j.status]||j.status)}</span></div><div class="progress"><i style="width:${j.progress*100}%"></i></div>${j.error?detailMarkup(j.id,'error','历史错误详情（不会影响新任务）',j.error,'error'):''}${detailMarkup(j.id,'logs','运行日志',j.logs.join('\n'))}<div class="downloads">${Object.keys(j.outputs).filter(k=>outputLabels[k]).map(k=>outputMarkup(j.id,k)).join('')}${runtimeInfo.local_open&&Object.keys(j.outputs).length?`<button onclick="openOutputFolder('${j.id}')">打开产物文件夹</button>`:''}</div><div class="job-actions">${actionMarkup(j)}</div></article>`).join('');
  box.querySelectorAll('details[data-open-key]').forEach(detail=>{detail.ontoggle=()=>{if(detail.open)openJobDetails.add(detail.dataset.openKey);else openJobDetails.delete(detail.dataset.openKey)};if(detail.open){const log=detail.querySelector('.logs');if(log)log.scrollTop=log.scrollHeight}})
}
$('#refresh-jobs').onclick=renderJobs; init().catch(e=>$('#health').textContent=e.message);
