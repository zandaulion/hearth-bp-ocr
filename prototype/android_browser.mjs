// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 zandaulion
// Operate only the authorized Android BP-reader tab through an ADB/SSH tunnel.
// Returns UI and OCR metadata; never reads or saves camera pixels.
import {parseArgs} from 'node:util';
const {values:options,positionals}=parseArgs({allowPositionals:true,options:{
  'cdp-port':{type:'string',default:'9333'},url:{type:'string',default:'http://127.0.0.1:8795/'},
  'report-prefix':{type:'string',default:'poco'}}});
const action=positionals[0]||'status',cdpPort=Number(options['cdp-port']);
if(positionals.length>1||!Number.isInteger(cdpPort)||cdpPort<1024||cdpPort>65535)throw new Error('Invalid action or CDP port');
const appUrl=new URL(options.url),prefix=options['report-prefix'];
if(!['http:','https:'].includes(appUrl.protocol)||!['127.0.0.1','localhost'].includes(appUrl.hostname)||!/^[a-z0-9-]+$/.test(prefix))throw new Error('Use a loopback app URL and a safe report prefix');
const controls={camera:'open-camera',capture:'take-photo',close:'close-camera'};
if(!['status','focus','refresh','diagnose','crop-diagnose','fallback-diagnose'].includes(action)&&!controls[action])throw new Error('Unknown Android test action');
const tabs=await(await fetch('http://127.0.0.1:'+cdpPort+'/json')).json();
const matching=tabs.filter(tab=>tab.type==='page'&&tab.url===appUrl.href);
if(matching.length!==1)throw new Error(`Expected one authorized BP-reader tab, got ${matching.length}`);
const socket=new WebSocket(matching[0].webSocketDebuggerUrl);
await new Promise((resolve,reject)=>{socket.addEventListener('open',resolve,{once:true});socket.addEventListener('error',reject,{once:true});});
if(['focus','refresh'].includes(action))await new Promise((resolve,reject)=>{
  const timer=setTimeout(()=>reject(new Error('Tab activation timed out')),5000);
  socket.addEventListener('message',({data})=>{const response=JSON.parse(data);if(response.id===0){clearTimeout(timer);response.error?reject(new Error(JSON.stringify(response.error))):resolve();}});
  socket.send(JSON.stringify({id:0,method:'Page.bringToFront'}));
});
const expression=`(async()=>{
  if(location.href!==${JSON.stringify(appUrl.href)})throw new Error('Wrong app tab');
  if(${JSON.stringify(action)}==='focus')return {action:'focus',visibility:document.visibilityState};
  if(${JSON.stringify(action)}!=='status'&&document.visibilityState!=='visible')throw new Error('Bring the BP-reader page to the foreground first');
  if(${JSON.stringify(action)}==='refresh'){
    const registration=await navigator.serviceWorker.getRegistration();
    await registration?.update();
    const installing=registration?.installing;
    if(installing&&installing.state!=='activated')await new Promise((resolve,reject)=>{
      const timer=setTimeout(()=>reject(new Error('Cache update timed out')),12000);
      installing.addEventListener('statechange',()=>{if(installing.state==='activated'){clearTimeout(timer);resolve();}else if(installing.state==='redundant'){clearTimeout(timer);reject(new Error('Cache install failed'));}});
    });
    const cachedVersions=await caches.keys();
    setTimeout(()=>location.reload(),100);
    return {action:'refresh',cachedVersions,reloading:true};
  }
  if(${JSON.stringify(action)}==='capture'&&document.getElementById('camera').srcObject?.getVideoTracks()[0]?.readyState!=='live')throw new Error('No live camera stream to capture');
  const buttonId=${JSON.stringify(controls[action]||null)};
  if(buttonId){const button=document.getElementById(buttonId);if(!button||button.disabled||button.hidden)throw new Error('Control unavailable');await button.onclick();}
  const until=Date.now()+15000;
  while(document.getElementById('result-status').textContent==='Reading…'&&Date.now()<until)await new Promise(resolve=>setTimeout(resolve,50));
  const track=document.getElementById('camera').srcObject?.getVideoTracks()[0],settings=track?.getSettings();
  const photo=document.getElementById('photo');
  let diagnosis=null;
  if(['diagnose','crop-diagnose','fallback-diagnose'].includes(${JSON.stringify(action)})){
    if(photo.hidden||track)throw new Error('Need an existing captured photo with camera stopped');
    const worker=new Worker(${JSON.stringify(action==='fallback-diagnose'?'./inference-worker.mjs?crop-fallback-1':'./inference-worker.mjs')},{type:'module'});
    try{
      await new Promise((resolve,reject)=>{
      const timer=setTimeout(()=>reject(new Error('Diagnostic replay timed out')),15000);
      worker.onerror=event=>{clearTimeout(timer);reject(new Error(event.message));};
      worker.onmessage=({data})=>{if(data.type==='ready'){clearTimeout(timer);resolve();}else if(data.type==='error'){clearTimeout(timer);reject(new Error(data.message));}};
      });
      const crops=${JSON.stringify(action)}==='crop-diagnose'?
        [{name:'center-wide',rect:[.15,.25,.7,.6]},{name:'center-tight',rect:[.25,.32,.6,.5]}]:[{name:'whole-photo',rect:[0,0,1,1]}];
      const replays=[];
      for(const crop of crops){
        const rect=crop.rect.map((v,i)=>Math.round(v*(i%2===0?photo.width:photo.height)));
        const bitmap=await createImageBitmap(photo,...rect);
        const output=await new Promise((resolve,reject)=>{
          const timer=setTimeout(()=>reject(new Error('Diagnostic replay timed out')),10000);
          worker.onmessage=({data})=>{if(data.type==='result'){clearTimeout(timer);resolve({elapsedMs:data.elapsedMs,result:data.result});}else if(data.type==='error'){clearTimeout(timer);reject(new Error(data.message));}};
          worker.postMessage({type:'read',id:1,bitmap},[bitmap]);
        });
        replays.push({crop:crop.name,rect,...output});
      }
      diagnosis=replays;
    }finally{worker.terminate();}
  }
  const modelConfig=await(await fetch('./models/config.json')).json();
  const cache=await caches.open('hearth-bp-'+modelConfig.version);
  return {at:new Date().toISOString(),action:${JSON.stringify(action)},modelConfig,offlineCropModuleCached:Boolean(await cache.match(new URL('./crop-fallback.mjs',location.href))),ready:document.getElementById('runtime').textContent==='Reader ready',busy:document.getElementById('result-status').textContent==='Reading…',
    status:document.getElementById('result-status').textContent,
    notice:document.getElementById('notice').textContent,
    elapsed:document.getElementById('result-detail').textContent,
    readings:Object.fromEntries(['sys','dia','pulse'].map(id=>[id,document.getElementById(id).value])),
    photoLabel:document.getElementById('photo-label').textContent,
    photoSize:!photo.hidden?{width:photo.width,height:photo.height}:null,
    camera:track?{state:track.readyState,width:settings.width,height:settings.height,frameRate:settings.frameRate,facingMode:settings.facingMode}:null,diagnosis};
})()`;
try{
  const response=await new Promise((resolve,reject)=>{
    const timer=setTimeout(()=>reject(new Error('Android browser operation timed out')),20000);
    socket.addEventListener('message',({data})=>{const message=JSON.parse(data);if(message.id===1){clearTimeout(timer);resolve(message);} });
    socket.send(JSON.stringify({id:1,method:'Runtime.evaluate',params:{expression,awaitPromise:true,returnByValue:true}}));
  });
  if(response.error||response.result?.exceptionDetails)throw new Error(JSON.stringify(response.error||response.result.exceptionDetails));
  const json=JSON.stringify(response.result.result.value,null,2);
  if(['diagnose','crop-diagnose','fallback-diagnose'].includes(action)){const {writeFile}=await import('node:fs/promises');const file={diagnose:prefix+'_current_photo_diagnosis.json','crop-diagnose':prefix+'_crop_diagnosis.json','fallback-diagnose':prefix+'_fallback_verification.json'}[action];await writeFile(new URL('./reports/'+file,import.meta.url),json+'\n');}
  console.log(json);
}finally{socket.close();}
