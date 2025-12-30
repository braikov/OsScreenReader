# Capture Service Architecture (C#)

## Responsibilities
- Run in background and listen for a global hotkey.
- On hotkey press, create a new capture session folder under the shared root.
- Capture the initial screenshot (baseline).
- Scan the screen by moving the mouse in fixed steps (e.g., 10px).
- For each mouse position, capture a screenshot and save it to the session folder.
- Write a session manifest to describe ordering and metadata.

## Session Folder Layout (created per hotkey press)

```
<root>/<session_id>/
  baseline.png
  frames/
    frame_000001.png
    frame_000002.png
    ...
  session.json
```

## Session Metadata (session.json)

```json
{
  "schema_version": "1.0",
  "session_id": "20240101T120000Z",
  "screen": { "width": 2560, "height": 1440, "dpi": 96 },
  "step": { "x": 10, "y": 10 },
  "baseline": "baseline.png",
  "frames_dir": "frames",
  "frames": [
    { "file": "frame_000001.png", "mouse": { "x": 0, "y": 0 } },
    { "file": "frame_000002.png", "mouse": { "x": 10, "y": 0 } }
  ]
}
```

## Notes
- The capture service does **not** compute diffs or OCR.
- The capture service only writes to disk; processing is deferred to the interactive region scanner.
