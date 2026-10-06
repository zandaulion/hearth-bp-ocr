// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 zandaulion
import {mkdir,copyFile,readFile,writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const pkgRoot=path.join(root,'node_modules/onnxruntime-web');
const dest=path.join(root,'web/vendor');
await mkdir(dest,{recursive:true});
for(const name of ['ort.wasm.min.mjs','ort-wasm-simd-threaded.mjs','ort-wasm-simd-threaded.wasm'])
  await copyFile(path.join(pkgRoot,'dist',name),path.join(dest,name));
const pkg=JSON.parse(await readFile(path.join(pkgRoot,'package.json'),'utf8'));
await copyFile(path.join(root,'licenses/ONNX-RUNTIME-LICENSE.txt'),path.join(dest,'ONNX-RUNTIME-LICENSE.txt'));
await writeFile(path.join(dest,'ONNX-RUNTIME-NOTICE.txt'),`ONNX Runtime Web ${pkg.version}\nLicense: ${pkg.license}\nSource: https://github.com/microsoft/onnxruntime\n`);
console.log(`Vendored ONNX Runtime Web ${pkg.version}; no photos copied.`);
