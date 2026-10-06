---
name: lenovo-android
description: Use Android phones attached to Lenovo through ADB over SSH, including the Galaxy A52 and Galaxy Z Fold4. Apply when asked to inspect these phones, open or test Android/browser apps, capture test results, or connect a local web app to a Lenovo-attached phone.
---

# Lenovo Android phones

Use the trusted SSH alias `lenovo`; ADB is installed there, not required locally.
The helper at [scripts/phones.py](scripts/phones.py) uses Python 3 standard library
and OpenSSH. Resolve its absolute path from this skill's directory. A project
Python executable is fine; don't install dependencies just to run it.

## Choose the device

Run `python <skill>/scripts/phones.py list` first. Known aliases:

| Alias | Model | Usual connection |
| --- | --- | --- |
| `a52` | Samsung SM_A525F | USB on Lenovo |
| `fold4` | Samsung SM_F936B | Wireless debugging on Lenovo |
| `poco` | Poco 24117RK2CG | Optional phone; only use when requested and present |

Every operation requires `--phone <alias-or-exact-current-serial>` and rediscovers
the device. Never substitute another phone when the requested one disconnects.
Wireless service names change after reconnection. If both phones are available
and the user hasn't selected one, inspect both read-only, then ask which to use
before changing their UI. A prior explicit choice remains valid during that task.

```powershell
python <skill>/scripts/phones.py --phone fold4 info
python <skill>/scripts/phones.py --phone a52 snapshot
python <skill>/scripts/phones.py --phone fold4 open 'http://127.0.0.1:8797/'
python <skill>/scripts/phones.py --phone fold4 tap-text 'Use camera'
```

`--dry-run` discovers/selects the actual device and prints a mutation plan without
applying it. Other commands: `tap x y`, `swipe x1 y1 x2 y2 duration_ms`, `key code`,
`foreground-chrome`, and `screenshot workspace-relative.png`. `key 224` wakes the
screen; it does not unlock a secured device. Use existing session authorization;
this skill does not grant permission for accounts, privacy prompts, or data changes.
If automatic approval review rejects camera permission, obtain the specific
device/origin authorization before retrying; don't tap through another route.

## Local web apps

Keep the desktop app bound to loopback. Once its server is running, use:

```powershell
python <skill>/scripts/phones.py --phone fold4 web --desktop-port 8765 --debug-browser
```

This runs a foreground SSH tunnel. Keep that exec session alive, then run `open`
in a separate call using the printed `phone_url`. Don't wait indefinitely on the
tunnel or launch visible helper windows. The default phone port is **8797**, distinct
from the app helper's historical 8795 default. Each alias gets separate Lenovo/desktop
forwarding ports, so A52 and Fold4 can be tested without overwriting each other's
connection. Conflicting mappings fail; choose another explicit port instead of
replacing an unrelated connection. Stop only sessions created for this task.

The optional Chrome debugger endpoint is printed. Read
[references/browser-testing.md](references/browser-testing.md) when controlling
browser pages or using the Hearth-specific test helper. Don't expose debug or app
ports publicly, clear Chrome data, or alter global phone networking to make a test work.

## Observe and verify

Take a fresh snapshot before UI actions and prefer unique text matches. Empty
Chrome accessibility trees are common: use the selected phone's debugger for
the intended app tab, or a fresh screenshot and measured coordinates. Avoid
repeating stale taps. On Fold4, folding, unfolding or rotation changes bounds;
recheck display size and UI rather than assuming the A52's dimensions.

Respect the task's photo-storage preferences. Screenshots can include camera
frames or private notifications; use metadata when enough, and save requested
screenshots only inside the current workspace. Read one test at a time, retain
failures, and distinguish a replay of one photo from a fresh capture. Camera
tests with an inaccessible phone establish preview access, not OCR accuracy.

For a disconnect, run `list` again, report the selected phone's state and await
reconnection if needed. For a backgrounded page, return to the authorized app/tab
once when appropriate; if the user is using another app, coordinate instead of
repeatedly stealing focus. Don't bypass a lock screen. After camera testing,
stop the stream unless the user is actively framing the next requested capture.
