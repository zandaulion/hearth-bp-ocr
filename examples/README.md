# Integration examples

These examples use the bundled exports and existing implementation. They are
AGPL-3.0-only, like the rest of Hearth. They are examples, not a separately
versioned SDK. See [integration instructions](../docs/INTEGRATION.md).

| File | Use |
| --- | --- |
| [browser-client.mjs](browser-client.mjs) | Persistent module-worker client with readiness, request IDs, one in-flight image, timeouts, bitmap ownership, disposal and the app's 1920-pixel image cap. Runs the complete browser pipeline. |
| [read_image.py](read_image.py) | Executable single-pass Python example that reads released thresholds from config. Does not run browser crop fallbacks. |

Browser usage after serving the worker/assets at `/hearth/`:

```js
import {createHearthReader} from './browser-client.mjs';

const reader = await createHearthReader('/hearth/inference-worker.mjs');
// file is a user-selected File; a canvas or ImageBitmap also works.
const {result, elapsedMs, imageSize} = await reader.read(file);
// Display result.status and request confirmation of any result.reading.
// Overlay result.rows in imageSize coordinates, not original-file coordinates.
// Keep reader for subsequent photos; dispose on component/page teardown.
reader.dispose();
```

Python usage after installing the Python dependencies from the root README:

```sh
python examples/read_image.py path/to/monitor.jpg
```

Use your own consented local photo; none is bundled. Exit code 0 means the script
ran, including a valid `retake` result. An unreadable file produces a nonzero
exit. JSON is printed to stdout; it is not saved. If you redirect stdout or
capture logs in your application, that creates your own persistence behavior.
