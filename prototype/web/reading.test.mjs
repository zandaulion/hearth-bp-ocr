// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 zandaulion
import test from 'node:test';
import assert from 'node:assert/strict';
import {assemble,decodeOutput,nms} from './reading.mjs';

function stack(values=[120,80,70],score=.9){
  return values.flatMap((value,index)=>{
    const y=20+index*70,text=String(value),x=80;
    return [{class:'row',box:[x-3,y-3,x+text.length*20+3,y+33],score},...Array.from(text,(digit,i)=>({class:digit,box:[x+i*20,y,x+i*20+16,y+30],score}))];
  });
}
test('assembles complete readings in left-to-right and top-to-bottom order',()=>{
  const result=assemble(stack().reverse());assert.equal(result.status,'candidate');assert.deepEqual(result.reading,{sys:120,dia:80,pulse:70});
});
test('missing digit does not become a guessed complete reading',()=>{
  const detections=stack();detections.pop();assert.equal(assemble(detections).reading,null);
});
test('duplicate hypotheses are suppressed across digit classes but not row/digit classes',()=>{
  const detections=stack();const d=detections.find(d=>d.class==='2');detections.push({...d,class:'8',score:.4});assert.equal(nms(detections).length,detections.length-1);assert.equal(assemble(detections).reading.sys,120);
});
test('multiple plausible stacks require a retake',()=>{
  const second=stack().map(d=>({...d,box:d.box.map((v,i)=>v+(i%2===0?300:0))}));assert.equal(assemble([...stack(),...second]).status,'retake');
});
test('low score keeps the candidate but requires review',()=>{
  assert.equal(assemble(stack([120,80,70],.5)).status,'review');
});
test('range checks never replace an observed digit with a plausible number',()=>{
  const result=assemble(stack([120,80,999]));assert.equal(result.status,'review');assert.equal(result.reading.pulse,999);
});
test('class index 2 is a row and class index 3 is digit 2',()=>{
  const data=new Float32Array(15*2);data[0]=20;data[1]=60;data[2]=30;data[3]=30;data[4]=10;data[5]=10;data[6]=20;data[7]=20;data[6*2]=.8;data[7*2+1]=.9;
  const detections=decodeOutput(data,[1,15,2],{scale:1,left:0,top:0,width:100,height:100});assert.deepEqual(detections.map(d=>d.class),['2','row']);
});
test('nonfinite scores and no detections cannot produce a reading',()=>{
  assert.equal(assemble([{class:'row',box:[0,0,10,10],score:NaN}]).reading,null);assert.equal(assemble([]).status,'retake');
});
test('a contained partial-stroke box cannot add a duplicate digit',()=>{
  const detections=stack(),d=detections.find(d=>d.class==='2');
  detections.push({...d,class:'1',score:.4,box:[d.box[0]+1,d.box[1]+1,d.box[0]+5,d.box[1]+12]});
  assert.equal(assemble(detections).reading.sys,120);
});
