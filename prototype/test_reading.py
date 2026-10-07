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
from prototype.reading import assemble,is_plausible_reading,nms
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

    def test_truncated_row_keeps_a_nearby_edge_digit(self):
        # Synthetic values; not sourced from a real reading.
        found=detections((123,78,64))
        next(d for d in found if d["class"]=="row")["box"][2]=114
        self.assertEqual(assemble(found)["reading"],{"sys":123,"dia":78,"pulse":64})

    def test_edge_recovery_does_not_sweep_a_distant_digit(self):
        found=detections((120,90,70));row=[d for d in found if d["class"]=="row"][1]
        found.append({"class":"1","box":[row["box"][2]+10,90,row["box"][2]+26,120],"score":.8})
        self.assertEqual(assemble(found)["reading"],{"sys":120,"dia":90,"pulse":70})

    def test_shared_plausibility_rule(self):
        self.assertTrue(is_plausible_reading({"sys":120,"dia":80,"pulse":70}))
        self.assertFalse(is_plausible_reading({"sys":12,"dia":80,"pulse":70}))
        self.assertFalse(is_plausible_reading(None))


if __name__=="__main__":unittest.main()
