# Interactive Region Scanner Architecture (Python)

## Responsibilities
- Watch the shared root folder for new session subfolders.
- For each session without a result JSON, process the frames sequentially.
- Compute diffs between consecutive frames (or baseline/previous frame).
- Extract changed regions and run OCR on those regions.
- Deduplicate overlapping regions and produce a single result JSON per session.

## Session Folder Layout (input)

```
<root>/<session_id>/
  baseline.png
  frames/
    frame_000001.png
    frame_000002.png
    ...
  session.json
```

## Output

```
<root>/<session_id>/
  result.json
```

## Result JSON (result.json)

```json
{
  "schema_version": "1.0",
  "session_id": "20240101T120000Z",
  "baseline": "baseline.png",
  "elements": [
    {
      "id": "elem_0001",
      "bbox": { "x": 100, "y": 200, "w": 80, "h": 24 },
      "text": "Connect to GitHub",
      "confidence": 0.91,
      "source": "hover-diff"
    }
  ]
}
```

## Processing Rules
- Only process sessions that **lack** `result.json`.
- Use diff-thresholding to detect changed pixels.
- Generate a bounding box for each contiguous changed region.
- Run OCR (EasyOCR) on the region with padding as needed.
- Merge overlapping boxes using IoU thresholding.
- Preserve `baseline.png` for training data export.
- Delete processed frame images after they are compared, keeping only the latest frame until the next diff is computed.
