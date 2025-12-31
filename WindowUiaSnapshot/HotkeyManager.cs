using System;
using System.Runtime.InteropServices;
using System.Windows.Forms;

namespace WindowUiaSnapshot;

/// <summary>
/// Manages registration and dispatch of a single global hotkey.
/// </summary>
internal sealed class HotkeyManager : NativeWindow, IDisposable
{
    private const int WmHotkey = 0x0312;
    private const int Id = 1;

    public event EventHandler? HotkeyPressed;

    public void RegisterHotkey(Keys key, bool ctrl, bool alt)
    {
        CreateHandle(new CreateParams());
        var modifiers = (ctrl ? ModControl : 0) | (alt ? ModAlt : 0);
        if (!RegisterHotKey(Handle, Id, modifiers, (int)key))
        {
            throw new InvalidOperationException("Failed to register global hotkey.");
        }
    }

    public void Unregister()
    {
        try
        {
            UnregisterHotKey(Handle, Id);
        }
        catch
        {
            // Ignore cleanup failures.
        }
    }

    protected override void WndProc(ref Message m)
    {
        if (m.Msg == WmHotkey && m.WParam.ToInt32() == Id)
        {
            HotkeyPressed?.Invoke(this, EventArgs.Empty);
        }

        base.WndProc(ref m);
    }

    public void Dispose()
    {
        Unregister();
        DestroyHandle();
    }

    private const int ModAlt = 0x0001;
    private const int ModControl = 0x0002;

    [DllImport("user32.dll", SetLastError = true)]
    private static extern bool RegisterHotKey(IntPtr hWnd, int id, int fsModifiers, int vk);

    [DllImport("user32.dll", SetLastError = true)]
    private static extern bool UnregisterHotKey(IntPtr hWnd, int id);
}
