# Browser tests on a Lenovo phone

The `web --debug-browser` command connects these private loopback paths:

```text
Phone 127.0.0.1:8797 → adb reverse → Lenovo alias-specific web port
  → SSH remote forwarding → desktop 127.0.0.1:<desktop-port>
Desktop alias-specific CDP port → SSH local forwarding
  → Lenovo alias-specific debug port → adb forward → Chrome local socket
```

| Phone | Lenovo web | Lenovo CDP | Desktop CDP |
| --- | ---: | ---: | ---: |
| a52 | 18875 | 19223 | 19333 |
| fold4 | 18876 | 19224 | 19334 |
| poco | 18877 | 19225 | 19335 |

Overrides are available through `web --help`. Keep endpoint selection associated
with the named device. Existing tunnels may target a different device; do not reuse them without checking.

Query `/json` on the printed CDP endpoint and retain only tabs matching the app
URL/origin relevant to the request. Bind operations to that specific tab; don't
inspect unrelated tabs. Native Android permission dialogs still require ADB UI
inspection and authorization, not a browser workaround.

The Chrome debugging socket exists only while Chrome is running on that phone.
If HTTP/CDP resets while ADB remains online, open the authorized app URL in Chrome
with the helper, then retry once. A valid port forward does not itself start Chrome.

Chrome may be foreground while a different tab is active. `Page.bringToFront`
selects the intended tab; `foreground-chrome` alone doesn't do that. Check
`document.visibilityState` before camera/replay work. Workers can suspend when
the page is backgrounded or the screen sleeps. A debugger timeout in that state
is a test-control interruption, not an OCR failure. Retry once after restoring
the intended tab; coordinate with the user if focus keeps changing.

## Hearth project

From this repository root, the app-specific helpers are:

- `prototype/android_device.py`: requires an explicit ADB serial; the skill helper
  adds named selection, discovery and correct nested quoting for URL parameters.
- `prototype/android_browser.mjs`: BP-reader controls and OCR metadata/replays.
  Its historical defaults target the old Poco connection, so pass the selected
  phone's endpoint, URL and report prefix explicitly. For Fold4 with the skill's
  default `web --debug-browser` tunnel:

  ```powershell
  node prototype/android_browser.mjs status --cdp-port 19334 --url http://127.0.0.1:8797/ --report-prefix fold4
  ```

  For A52 use CDP port 19333 and prefix `a52`. Other actions: `focus`, `camera`,
  `capture`, `close`, `refresh`, `diagnose`, `crop-diagnose`, `fallback-diagnose`.
  Camera actions require user authorization and readiness. Never reuse a default
  endpoint merely because its socket happens to respond.
- `prototype/serve.py`: static BP reader on desktop loopback 8765.
- `prototype/android_test_server.py`: sample timing harness, default desktop 8875.
  Give each phone a distinct report output; its historical offline filename is
  A52-specific, so don't overwrite A52 results with a different device's run.

Use the installed phone skill for general Android tests, not only blood-pressure
OCR. App-specific scripted expressions belong with their app's test helpers.
Observe source/ground-truth values independently when scoring live OCR. Keep
model/config fingerprints, and don't count crop replays or repeated training
samples as independent evidence of the precision target.
