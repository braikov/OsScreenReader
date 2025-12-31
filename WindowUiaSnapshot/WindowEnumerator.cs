using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using System.Windows.Automation;
using WindowUiaSnapshot.Models;

namespace WindowUiaSnapshot;

/// <summary>
/// Enumerates top-level visible windows and applies basic filtering.
/// </summary>
internal sealed class WindowEnumerator
{
    /// <summary>
    /// Enumerate visible top-level windows that pass basic filters.
    /// </summary>
    public IReadOnlyList<WindowSnapshot> EnumerateVisibleWindows()
    {
        var results = new List<WindowSnapshot>();
        EnumWindows((hwnd, _) =>
        {
            if (!IsWindowVisible(hwnd))
            {
                return true;
            }

            if (!GetWindowRect(hwnd, out var rect))
            {
                return true;
            }

            var width = rect.right - rect.left;
            var height = rect.bottom - rect.top;
            if (width <= 1 || height <= 1)
            {
                return true;
            }

            var title = GetWindowTextSafe(hwnd);
            var pid = GetProcessId(hwnd);

            string? uiaName = null;
            if (string.IsNullOrWhiteSpace(title))
            {
                uiaName = TryGetUiaName(hwnd);
                if (string.IsNullOrWhiteSpace(uiaName))
                {
                    return true;
                }
            }

            results.Add(new WindowSnapshot
            {
                Hwnd = $"0x{hwnd.ToInt64():X}",
                Title = string.IsNullOrWhiteSpace(title) ? uiaName ?? string.Empty : title,
                ProcessId = pid,
                Rect = new RectInt
                {
                    Left = rect.left,
                    Top = rect.top,
                    Right = rect.right,
                    Bottom = rect.bottom,
                },
                TimestampUtc = DateTime.UtcNow,
            });

            return true;
        }, IntPtr.Zero);

        return results;
    }

    private static string GetWindowTextSafe(IntPtr hwnd)
    {
        var length = GetWindowTextLength(hwnd);
        if (length == 0)
        {
            return string.Empty;
        }

        var sb = new StringBuilder(length + 1);
        _ = GetWindowText(hwnd, sb, sb.Capacity);
        return sb.ToString().Trim();
    }

    private static int GetProcessId(IntPtr hwnd)
    {
        _ = GetWindowThreadProcessId(hwnd, out var pid);
        return pid;
    }

    private static string? TryGetUiaName(IntPtr hwnd)
    {
        try
        {
            var element = AutomationElement.FromHandle(hwnd);
            return element?.Current.Name?.Trim();
        }
        catch
        {
            return null;
        }
    }

    private delegate bool EnumWindowsProc(IntPtr hwnd, IntPtr lParam);

    [DllImport("user32.dll")]
    private static extern bool EnumWindows(EnumWindowsProc lpEnumFunc, IntPtr lParam);

    [DllImport("user32.dll")]
    private static extern bool IsWindowVisible(IntPtr hWnd);

    [DllImport("user32.dll", SetLastError = true)]
    private static extern int GetWindowText(IntPtr hWnd, StringBuilder lpString, int nMaxCount);

    [DllImport("user32.dll", SetLastError = true)]
    private static extern int GetWindowTextLength(IntPtr hWnd);

    [DllImport("user32.dll")]
    private static extern bool GetWindowRect(IntPtr hWnd, out NativeRect lpRect);

    [DllImport("user32.dll")]
    private static extern uint GetWindowThreadProcessId(IntPtr hWnd, out int lpdwProcessId);

    [StructLayout(LayoutKind.Sequential)]
    private struct NativeRect
    {
        public int left;
        public int top;
        public int right;
        public int bottom;
    }
}
