// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 zandaulion
const CACHE='hearth-bp-v2-adaptive2';
const ASSETS=['./','./index.html','./styles.css','./app.mjs','./reading.mjs','./crop-fallback.mjs','./adaptive-crop.mjs','./inference-worker.mjs','./manifest.webmanifest','./icon.svg','./icon-192.png','./icon-512.png','./vendor/ort.wasm.min.mjs','./vendor/ort-wasm-simd-threaded.mjs','./vendor/ort-wasm-simd-threaded.wasm','./models/bp-detector.onnx','./models/bp-digits.onnx','./models/config.json'];
self.addEventListener('install',event=>event.waitUntil(caches.open(CACHE).then(cache=>cache.addAll(ASSETS)).then(()=>self.skipWaiting())));
self.addEventListener('activate',event=>event.waitUntil(caches.keys().then(keys=>Promise.all(keys.filter(key=>key.startsWith('hearth-bp-')&&key!==CACHE).map(key=>caches.delete(key)))).then(()=>self.clients.claim())));
self.addEventListener('fetch',event=>{
  if(event.request.method!=='GET'||new URL(event.request.url).origin!==self.location.origin)return;
  event.respondWith(caches.open(CACHE).then(async cache=>{
    const saved=await cache.match(event.request);if(saved)return saved;
    const response=await fetch(event.request);return response;
  }));
});
