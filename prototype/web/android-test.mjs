// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 zandaulion
const $=id=>document.getElementById(id);
const started=performance.now(),params=new URLSearchParams(location.search);
let report={kind:'hearth-android-test',startedAt:new Date().toISOString(),phase:params.get('phase')||location.hash.slice(1)||'online',userAgent:navigator.userAgent,hardwareConcurrency:navigator.hardwareConcurrency,deviceMemoryGB:navigator.deviceMemory??null,secureContext:isSecureContext,capabilities:{worker:typeof Worker!=='undefined',offscreenCanvas:typeof OffscreenCanvas!=='undefined',webAssembly:typeof WebAssembly!=='undefined',camera:!!navigator.mediaDevices?.getUserMedia,serviceWorker:'serviceWorker'in navigator},viewport:{width:document.documentElement.clientWidth,height:document.documentElement.clientHeight,pixelRatio:devicePixelRatio},runs:[]};
let worker,stream,nextId=0;
const pending=new Map();
function show(message){$('status').textContent=message;$('output').textContent=JSON.stringify(report,null,2);}
async function save(){show($('status').textContent);try{const response=await fetch('./android-test-result',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(report)});report.metricsSaved=response.ok;}catch{report.metricsSaved=false;}localStorage.setItem('hearth-android-test-'+report.phase,JSON.stringify(report));}
function read(bitmap){return new Promise((resolve,reject)=>{const id=++nextId;const timer=setTimeout(()=>{pending.delete(id);reject(new Error('OCR exceeded 30 seconds'));},30000);pending.set(id,{resolve,reject,timer});worker.postMessage({type:'read',id,bitmap},[bitmap]);});}
function stats(values){const sorted=[...values].sort((a,b)=>a-b);return {count:values.length,minMs:sorted[0],medianMs:sorted[Math.floor(sorted.length/2)],p95Ms:sorted[Math.ceil(sorted.length*.95)-1],maxMs:sorted.at(-1)};}
async function main(){
  try{
    if(report.phase==='collect'){
      const saved=localStorage.getItem('hearth-android-test-offline');
      if(!saved)throw new Error('No offline result is available');
      report=JSON.parse(saved);show('Recovering the completed offline result.');await save();return;
    }
    report.modelConfig=await(await fetch('./models/config.json')).json();
    worker=new Worker('./inference-worker.mjs',{type:'module'});
    await new Promise((resolve,reject)=>{
      const timer=setTimeout(()=>reject(new Error('Model initialization exceeded 90 seconds')),90000);
      worker.onmessage=({data})=>{
        if(data.type==='ready'){clearTimeout(timer);resolve();return;}
        const request=pending.get(data.id);
        if(request){clearTimeout(request.timer);pending.delete(data.id);data.type==='result'?request.resolve(data):request.reject(new Error(data.message));}
        else if(data.type==='error'){clearTimeout(timer);reject(new Error(data.message));}
      };
      worker.onerror=event=>{clearTimeout(timer);reject(new Error(event.message||'Worker failed'));};
    });
    report.modelReadyMs=performance.now()-started;
    const response=await fetch('./samples/test-manifest.json');
    if(!response.ok)throw new Error('Provide a local consented test manifest; see README.md');
    const samples=await response.json();
    if(!Array.isArray(samples)||!samples.length)throw new Error('Test manifest must contain at least one sample');
    const images=[];
    for(const sample of samples)images.push({sample,blob:await(await fetch('./samples/'+sample.file)).blob()});
    for(let repeat=0;repeat<6;repeat++)for(const {sample,blob} of images){
      show(`Reading ${sample.file} · pass ${repeat+1}/6…`);
      const bitmap=await createImageBitmap(blob),t=performance.now();
      const output=await read(bitmap),roundTripMs=performance.now()-t;
      const exact=['sys','dia','pulse'].every(key=>output.result.reading?.[key]===sample.expected[key]);
      report.runs.push({file:sample.file,datasetRole:sample.role,repeat,expected:sample.expected,reading:output.result.reading,status:output.result.status,exact,workerMs:output.elapsedMs,roundTripMs});
    }
    report.ocrSummary={correct:report.runs.filter(r=>r.exact).length,total:report.runs.length,warmWorker:stats(report.runs.filter(r=>r.repeat>0).map(r=>r.workerMs)),warmRoundTrip:stats(report.runs.filter(r=>r.repeat>0).map(r=>r.roundTripMs))};
    report.viewport={width:document.documentElement.clientWidth,height:document.documentElement.clientHeight,pixelRatio:devicePixelRatio};
    report.horizontalOverflow=document.documentElement.scrollWidth>document.documentElement.clientWidth;
    if('serviceWorker'in navigator){await navigator.serviceWorker.register('./android-test-sw.js');await navigator.serviceWorker.ready;report.serviceWorkerReady=true;}
    report.completedAt=new Date().toISOString();show('Device OCR checks completed.');await save();
  }catch(error){report.error=error.message;report.completedAt=new Date().toISOString();show('Device check failed: '+error.message);await save();}
}
$('camera').onclick=async()=>{
  try{stream?.getTracks().forEach(t=>t.stop());stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'}},audio:false});$('video').srcObject=stream;$('video').hidden=false;await $('video').play();$('stop').hidden=false;const s=stream.getVideoTracks()[0].getSettings();report.cameraTest={passed:true,settings:{width:s.width,height:s.height,frameRate:s.frameRate,facingMode:s.facingMode}};show('Camera preview is running.');}
  catch(error){report.cameraTest={passed:false,error:error.name+': '+error.message};show('Camera test: '+error.message);}
  await save();
};
$('stop').onclick=()=>{stream?.getTracks().forEach(t=>t.stop());stream=null;$('video').hidden=true;$('stop').hidden=true;show('Camera stopped.');};
main();
