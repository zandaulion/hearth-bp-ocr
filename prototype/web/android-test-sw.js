// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 zandaulion
// Harness revision 2. Development origin only: use the production offline strategy and also cache
// the device harness. The normal PWA registers sw.js, not this test worker.
importScripts('./sw.js');
ASSETS.push('./android-test.html','./android-test.mjs');
