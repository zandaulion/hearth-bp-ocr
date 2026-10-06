# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 zandaulion
"""Check geometry/rejection behavior and Python/JavaScript agreement."""
import json
from pathlib import Path
import subprocess
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prototype.reading import assemble,nms
from prototype.detector import decode_output,letterbox


def detections(values=(120,80,70),score=.9):
    result=[]
    for i,value in enumerate(values):
        y=20+i*70;x=80;text=str(value)
        result.append({"class":"row","box":[x-3,y-3,x+len(text)*20+3,y+33],"score":score})
        result.extend({"class":digit,"box":[x+j*20,y,x+j*20+16,y+30],"score":score} for j,digit in enumerate(text))
    return result


class ReadingTests(unittest.TestCase):
    def test_missing_digit_refuses(self):
        self.assertIsNone(assemble(detections()[:-1])["reading"])

    def test_range_check_does_not_change_values(self):
        result=assemble(detections((120,80,999)))
        self.assertEqual(result["status"],"review")
        self.assertEqual(result["reading"]["pulse"],999)

    def test_letterbox_rgb_and_padding(self):
        image=np.zeros((50,100,3),np.uint8);image[:]=[10,20,30]
        tensor,t=letterbox(image,100)
        self.assertEqual(tensor.shape,(1,3,100,100));self.assertEqual(t["top"],25)
        np.testing.assert_allclose(tensor[0,:,40,40],np.array([30,20,10])/255,atol=1e-7)
        np.testing.assert_allclose(tensor[0,:,0,0],np.array([114,114,114])/255,atol=1e-7)

    def test_decode_class_mapping_and_unpadding(self):
        out=np.zeros((1,15,1),np.float32);out[0,:4,0]=[50,50,20,30];out[0,7,0]=.9
        result=decode_output(out,{"scale":.5,"left":0,"top":25,"width":200,"height":100})
        self.assertEqual(result[0]["class"],"2")
        self.assertEqual(result[0]["box"],[80,20,120,80])

    def test_python_javascript_agree_on_rejection_and_values(self):
        fixtures=[detections(),detections()[:-1],detections((120,80,999)),detections(score=.5),[]]
        other=[{**d,"box":[v+(300 if i%2==0 else 0) for i,v in enumerate(d["box"])]} for d in detections()]
        fixtures.append(detections()+other)
        script="import {assemble} from './web/reading.mjs'; let chunks=''; for await(const chunk of process.stdin) chunks+=chunk; console.log(JSON.stringify(JSON.parse(chunks).map(d=>{const r=assemble(d);return {status:r.status,reading:r.reading};})));"
        output=subprocess.run(["node","--input-type=module","-e",script],input=json.dumps(fixtures),text=True,capture_output=True,cwd=ROOT / "prototype",check=True)
        expected=[{k:assemble(f)[k] for k in ("status","reading")} for f in fixtures]
        self.assertEqual(json.loads(output.stdout),expected)


if __name__=="__main__":unittest.main()
