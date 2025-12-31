using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace WindowUiaSnapshot;

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
