# WindowUiaSnapshot Architecture

## Purpose
Tray-based WinForms app that listens for Ctrl+Alt+W and writes UI Automation snapshots of all visible top-level windows to `%LOCALAPPDATA%\WindowUiaSnapshot\snapshots\snapshot_yyyyMMdd_HHmmss.json`.

## Components
- `Program`: entry point; runs `TrayAppContext`.
- `TrayAppContext`: hosts `NotifyIcon`, context menu, hotkey registration, snapshot orchestration on an STA background thread, and balloon notifications.
- `HotkeyManager`: wraps `RegisterHotKey`/`UnregisterHotKey` and raises an event on Ctrl+Alt+W.
- `WindowEnumerator`: P/Invoke `EnumWindows`/`IsWindowVisible`/`GetWindowRect`/`GetWindowText` and basic filters (visible, size > 1px, has title or UIA name).
- `UiaSnapshotter`: builds a UIA tree with limits (`maxDepth=8`, `maxChildren=200`, `timeout=1500ms`) using `AutomationElement` and `TreeWalker.RawViewWalker`; tolerant of UIA failures.
- `SnapshotWriter`: serializes `SnapshotRoot` to JSON (indented) in the snapshots folder.
- Models (`Models/*.cs`): DTOs for snapshot, windows, rectangles, and UIA nodes.

## Data Flow
1. Hotkey/menu triggers snapshot.
2. `WindowEnumerator` collects window metadata (hwnd, title, pid, rect).
3. For each window, `UiaSnapshotter` attempts to build the UIA tree; errors are captured per window.
4. `SnapshotWriter` writes a JSON file including machine/user/os metadata and all window entries.

## Notes
- Built as `net8.0-windows`, `Microsoft.WindowsDesktop.App` framework reference provides UIA assemblies.
- No external NuGet dependencies.
- Cleans up hotkey and tray icon on exit.
