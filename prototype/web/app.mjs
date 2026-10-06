// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 zandaulion
const $=id=>document.getElementById(id);
const controls=[$('open-camera'),$('choose-photo'),...document.querySelectorAll('[data-sample]')];
let worker,ready=false,busy=false,currentPhoto,originalPhoto,originalLabel,stream,result,requestId=0,cropping=false,cropBox,startPoint;

function setBusy(value){busy=value;controls.forEach(b=>b.disabled=value||!ready);$('read-again').disabled=value;$('crop').disabled=value;$('use-crop').disabled=value;$('restore-photo').disabled=value;}
function notice(text,tone=''){$('notice').textContent=text;$('notice').className=`notice ${tone}`;}
function clearReading(){result=null;for(const key of ['sys','dia','pulse']){$(key).value='';$(key).disabled=true;}$('verified').checked=false;$('verified').disabled=true;$('copy').disabled=true;$('copy').textContent='Copy verified reading';$('result-detail').textContent='';}
function updateCopy(){const valid=['sys','dia','pulse'].every(key=>/^\d{1,3}$/.test($(key).value));$('copy').disabled=!valid||!$('verified').checked||busy;}
function showError(message){setBusy(false);$('result-status').textContent='Please try again';$('result-status').className='pill review';notice(message,'error');}

worker=new Worker('./inference-worker.mjs',{type:'module'});
worker.onmessage=({data})=>{
  if(data.type==='ready'){ready=true;$('runtime').textContent='Reader ready';setBusy(false);return;}
  if(data.id!==undefined&&data.id!==requestId)return;
  if(data.type==='error'){
    if(!ready)$('runtime').textContent='Reader unavailable';
    showError(!ready?`${data.message}. Refresh the app when its local files are available.`:`Couldn't read this photo. ${data.message}`);return;
  }
  if(data.type!=='result')return;
  result=data.result;setBusy(false);paintPhoto();
  $('result-status').textContent=result.status==='retake'?'Retake photo':result.status==='review'?'Please review':'Check values';
  $('result-status').className=`pill ${result.status==='candidate'?'active':'review'}`;
  if(result.reading){
    for(const key of ['sys','dia','pulse']){$(key).value=result.reading[key];$(key).disabled=false;}
    $('verified').disabled=false;
    notice(result.status==='candidate'?'Check every value against the display. You can correct a number before copying.':'Part of this reading is uncertain. Check every number carefully or take another photo.',result.status==='review'?'warning':'');
  }else{clearReading();notice('We couldn’t find a complete three-row reading. Crop around the display or try a clearer, upright photo.','warning');}
  $('result-detail').textContent=`Read locally in ${(data.elapsedMs/1000).toFixed(2)} seconds. Nothing is saved or uploaded.`;
};
worker.onerror=event=>{ready=false;$('runtime').textContent='Reader unavailable';showError(`The reader couldn't start. Refresh this page. ${event.message||''}`);};

function paintPhoto(){
  if(!currentPhoto)return;
  const canvas=$('photo'),ctx=canvas.getContext('2d');
  canvas.classList.toggle('cropping',cropping);
  canvas.width=currentPhoto.width;canvas.height=currentPhoto.height;ctx.drawImage(currentPhoto,0,0);
  if(cropping&&cropBox){
    const [x,y,w,h]=cropBox;ctx.fillStyle='rgba(15,46,39,.35)';ctx.fillRect(0,0,canvas.width,canvas.height);ctx.clearRect(x,y,w,h);ctx.drawImage(currentPhoto,x,y,w,h,x,y,w,h);ctx.strokeStyle='#dcebbb';ctx.lineWidth=Math.max(2,canvas.width/250);ctx.strokeRect(x,y,w,h);
  }else if(result?.rows){
    for(const [i,row] of result.rows.entries()){
      const [x1,y1,x2,y2]=row.box;ctx.strokeStyle='#9eca6b';ctx.lineWidth=Math.max(2,canvas.width/260);ctx.strokeRect(x1,y1,x2-x1,y2-y1);
      ctx.font=`600 ${Math.max(12,canvas.width/38)}px system-ui`;ctx.fillStyle='#164b43';
      const label=['SYS','DIA','PULSE'][i],size=ctx.measureText(label).width;
      const y=Math.max(0,y1-Math.max(18,canvas.width/30));ctx.fillRect(x1,y,size+12,Math.max(18,canvas.width/30));ctx.fillStyle='#fff';ctx.fillText(label,x1+6,y+Math.max(14,canvas.width/40));
    }
  }
}

async function readPhoto(){
  if(!ready||busy||!currentPhoto)return;
  clearReading();setBusy(true);requestId++;$('result-status').textContent='Reading…';$('result-status').className='pill';notice('Reading the display on your device…');
  try{const bitmap=await createImageBitmap(currentPhoto);worker.postMessage({type:'read',id:requestId,bitmap},[bitmap]);}
  catch(error){showError(`Couldn't prepare the photo: ${error.message}`);}
}

async function loadPhoto(source,label){
  if(busy||!ready)return;
  clearReading();setBusy(true);
  try{
    let bitmap=await createImageBitmap(source,{imageOrientation:'from-image'});
    const scale=Math.min(1,1920/Math.max(bitmap.width,bitmap.height));
    if(scale<1){const resized=await createImageBitmap(bitmap,{resizeWidth:Math.round(bitmap.width*scale),resizeHeight:Math.round(bitmap.height*scale),resizeQuality:'high'});bitmap.close();bitmap=resized;}
    currentPhoto?.close();originalPhoto?.close();currentPhoto=bitmap;originalPhoto=await createImageBitmap(bitmap);
    originalLabel=label;$('restore-photo').hidden=true;
    closeCamera();stopCrop();$('empty').hidden=true;$('photo').hidden=false;$('photo-label').hidden=false;$('photo-label').textContent=label;$('edit-actions').hidden=false;
    paintPhoto();setBusy(false);await readPhoto();
  }catch(error){showError(`Couldn't open that image. Try a JPEG, PNG or WebP photo. ${error.message}`);}
}

$('choose-photo').onclick=()=>$('file').click();
$('file').onchange=async()=>{const file=$('file').files[0];if(file)await loadPhoto(file,file.name);$('file').value='';};
for(const button of document.querySelectorAll('[data-sample]'))button.onclick=async()=>{
  if(busy)return;
  try{const response=await fetch(`./samples/${button.dataset.sample}`);if(!response.ok)throw new Error('Sample unavailable');await loadPhoto(await response.blob(),button.textContent);}
  catch(error){showError(error.message);}
};
$('read-again').onclick=()=>{stopCrop();readPhoto();};

function closeCamera(){stream?.getTracks().forEach(track=>track.stop());stream=null;$('camera').srcObject=null;$('camera').hidden=true;$('camera-actions').hidden=true;$('open-camera').textContent='Use camera';$('photo').hidden=!currentPhoto;$('empty').hidden=Boolean(currentPhoto);$('edit-actions').hidden=!currentPhoto;}
$('open-camera').onclick=async()=>{
  if(!navigator.mediaDevices?.getUserMedia){showError('Camera access requires HTTPS or localhost. Choose a photo instead.');return;}
  try{
    closeCamera();
    stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'},width:{ideal:1920},height:{ideal:1080}},audio:false});
    clearReading();stopCrop();$('camera').srcObject=stream;await $('camera').play();$('camera').hidden=false;$('photo').hidden=true;$('empty').hidden=true;$('photo-label').hidden=true;$('camera-actions').hidden=false;$('edit-actions').hidden=true;
    $('result-status').textContent='Camera open';notice('Hold the display upright and fill the frame. Avoid reflections.');
  }catch(error){closeCamera();showError(`Couldn't access the camera. Check permission or choose a photo. ${error.message}`);}
};
$('close-camera').onclick=()=>{closeCamera();$('photo').hidden=!currentPhoto;$('empty').hidden=Boolean(currentPhoto);$('edit-actions').hidden=!currentPhoto;};
$('take-photo').onclick=async()=>{
  const video=$('camera');if(!video.videoWidth||!video.videoHeight)return;
  const canvas=document.createElement('canvas');canvas.width=video.videoWidth;canvas.height=video.videoHeight;canvas.getContext('2d').drawImage(video,0,0);
  closeCamera();await loadPhoto(canvas,'Camera capture');
};
document.addEventListener('visibilitychange',()=>{if(document.hidden)closeCamera();});

function stopCrop(){cropping=false;cropBox=null;startPoint=null;$('photo').classList.remove('cropping');$('use-crop').hidden=true;$('cancel-crop').hidden=true;$('crop').hidden=false;$('capture-hint').textContent='Three-row home monitors work best. You can crop a photo to make the display easier to read.';}
$('crop').onclick=()=>{
  if(!currentPhoto||busy)return;cropping=true;cropBox=null;clearReading();result=null;$('result-status').textContent='Select display';notice('Drag around all three readings, then choose Use crop.');$('crop').hidden=true;$('cancel-crop').hidden=false;$('capture-hint').textContent='Drag a rectangle around the display, including all three readings, then choose Use crop.';paintPhoto();
};
function point(event){const rect=$('photo').getBoundingClientRect();return [Math.max(0,Math.min(currentPhoto.width,(event.clientX-rect.left)*currentPhoto.width/rect.width)),Math.max(0,Math.min(currentPhoto.height,(event.clientY-rect.top)*currentPhoto.height/rect.height))];}
$('photo').onpointerdown=event=>{if(!cropping)return;startPoint=point(event);$('photo').setPointerCapture(event.pointerId);};
$('photo').onpointermove=event=>{if(!cropping||!startPoint)return;const end=point(event);cropBox=[Math.min(startPoint[0],end[0]),Math.min(startPoint[1],end[1]),Math.abs(end[0]-startPoint[0]),Math.abs(end[1]-startPoint[1])];paintPhoto();};
$('photo').onpointerup=()=>{if(!cropping)return;startPoint=null;$('use-crop').hidden=!cropBox||cropBox[2]<20||cropBox[3]<20;};
$('photo').onpointercancel=()=>{startPoint=null;};
$('cancel-crop').onclick=()=>{stopCrop();paintPhoto();readPhoto();};
$('use-crop').onclick=async()=>{
  if(!cropBox||busy)return;
  try{const [x,y,w,h]=cropBox.map(Math.round);const cropped=await createImageBitmap(currentPhoto,x,y,w,h);currentPhoto.close();currentPhoto=cropped;stopCrop();$('restore-photo').hidden=false;$('photo-label').textContent='Cropped display';paintPhoto();await readPhoto();}
  catch(error){stopCrop();showError(`Couldn't crop the photo: ${error.message}`);}
};
$('restore-photo').onclick=async()=>{
  if(!originalPhoto||busy)return;
  try{const restored=await createImageBitmap(originalPhoto);currentPhoto.close();currentPhoto=restored;stopCrop();$('restore-photo').hidden=true;$('photo-label').textContent=originalLabel;paintPhoto();await readPhoto();}
  catch(error){showError(`Couldn't restore the photo: ${error.message}`);}
};

for(const key of ['sys','dia','pulse'])$(key).oninput=()=>{$('verified').checked=false;updateCopy();};
$('verified').onchange=updateCopy;
$('copy').onclick=async()=>{
  updateCopy();if($('copy').disabled)return;
  const text=`SYS ${$('sys').value} mmHg\nDIA ${$('dia').value} mmHg\nPulse ${$('pulse').value} bpm`;
  try{await navigator.clipboard.writeText(text);$('copy').textContent='Verified reading copied';}
  catch{notice('Clipboard access is unavailable. You can select and copy the checked values above.','warning');}
};

if('serviceWorker'in navigator&&isSecureContext){
  navigator.serviceWorker.register('./sw.js').then(()=>navigator.serviceWorker.ready).then(()=>{$('offline-status').textContent='App and reader are cached for offline use.';}).catch(()=>{$('offline-status').textContent='Offline caching unavailable. Keep this local app open.';});
}else $('offline-status').textContent='Offline installation needs HTTPS or localhost.';
