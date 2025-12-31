using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using WindowUiaSnapshot.Models;

namespace WindowUiaSnapshot;

/// <summary>
/// Application context hosting the tray icon, menu, and snapshot orchestration.
/// </summary>
internal sealed class TrayAppContext : ApplicationContext
{
    private readonly NotifyIcon _notifyIcon;
    private readonly HotkeyManager _hotkeyManager;
    private bool _isSnapshotInProgress;

    public TrayAppContext()
    {
        _notifyIcon = BuildNotifyIcon();
        _hotkeyManager = new HotkeyManager();
        _hotkeyManager.HotkeyPressed += (_, _) => TriggerSnapshot();
        _hotkeyManager.RegisterHotkey(Keys.W, ctrl: true, alt: true);
    }

    /// <summary>
    /// Create tray icon with context menu items.
    /// </summary>
    private NotifyIcon BuildNotifyIcon()
    {
        var menu = new ContextMenuStrip();
        var snapshotItem = new ToolStripMenuItem("Snapshot now", null, (_, _) => TriggerSnapshot());
        var openFolderItem = new ToolStripMenuItem("Open output folder", null, (_, _) => OpenOutputFolder());
        var exitItem = new ToolStripMenuItem("Exit", null, (_, _) => ExitThread());
        menu.Items.AddRange(new ToolStripItem[] { snapshotItem, openFolderItem, exitItem });

        return new NotifyIcon
        {
            Icon = SystemIcons.Application,
            Visible = true,
            Text = "Window UIA Snapshot",
            ContextMenuStrip = menu,
        };
    }

    /// <summary>
    /// Open the snapshot output directory in Explorer.
    /// </summary>
    private void OpenOutputFolder()
    {
        try
        {
            var writer = new SnapshotWriter();
            var dir = writer.EnsureOutputDirectory();
            _ = System.Diagnostics.Process.Start("explorer.exe", dir);
        }
        catch (Exception ex)
        {
            ShowBalloon($"Failed to open folder: {ex.Message}");
        }
    }

    /// <summary>
    /// Fire a snapshot if one is not already running.
    /// </summary>
    private void TriggerSnapshot()
    {
        if (_isSnapshotInProgress)
        {
            return;
        }

        _isSnapshotInProgress = true;
        ShowBalloon("Collecting window snapshots...");
        _ = RunSnapshotAsync();
    }

    /// <summary>
    /// Run the snapshot logic on an STA background thread to allow UIA calls.
    /// </summary>
    private async Task RunSnapshotAsync()
    {
        try
        {
            var tcs = new TaskCompletionSource();
            var thread = new Thread(() =>
            {
                try
                {
                    SnapshotOnce();
                    tcs.SetResult();
                }
                catch (Exception ex)
                {
                    tcs.SetException(ex);
                }
            });
            thread.SetApartmentState(ApartmentState.STA);
            thread.IsBackground = true;
            thread.Start();
            await tcs.Task.ConfigureAwait(false);
            ShowBalloon("Snapshot saved.");
        }
        catch (Exception ex)
        {
            ShowBalloon($"Snapshot failed: {ex.Message}");
        }
        finally
        {
            _isSnapshotInProgress = false;
        }
    }

    /// <summary>
    /// Perform a single snapshot of all visible windows.
    /// </summary>
    private void SnapshotOnce()
    {
        var enumerator = new WindowEnumerator();
        var uia = new UiaSnapshotter();
        var writer = new SnapshotWriter();

        var windows = enumerator.EnumerateVisibleWindows();
        var root = new SnapshotRoot();
        foreach (var window in windows)
        {
            var handle = ParseHandle(window.Hwnd);
            var (node, error) = uia.SnapshotAsync(handle).GetAwaiter().GetResult();
            window.Uia = node;
            window.UiaError = error;
            root.Windows.Add(window);
        }

        writer.WriteAsync(root).GetAwaiter().GetResult();
    }

    /// <summary>
    /// Parse a hex hwnd string into an IntPtr.
    /// </summary>
    private static IntPtr ParseHandle(string hwndHex)
    {
        try
        {
            return new IntPtr(Convert.ToInt64(hwndHex, 16));
        }
        catch
        {
            return IntPtr.Zero;
        }
    }

    /// <summary>
    /// Display a balloon notification from the tray icon.
    /// </summary>
    private void ShowBalloon(string message)
    {
        _notifyIcon.BalloonTipTitle = "Window UIA Snapshot";
        _notifyIcon.BalloonTipText = message;
        _notifyIcon.ShowBalloonTip(3000);
    }

    protected override void ExitThreadCore()
    {
        _hotkeyManager.Dispose();
        _notifyIcon.Visible = false;
        _notifyIcon.Dispose();
        base.ExitThreadCore();
    }
}
