using System;
using System.Collections.Generic;

namespace WindowUiaSnapshot;

internal sealed class SnapshotRoot
{
    public string MachineName { get; set; } = Environment.MachineName;
    public string UserName { get; set; } = Environment.UserName;
    public string OsVersion { get; set; } = Environment.OSVersion.ToString();
    public string Hotkey { get; set; } = "Ctrl+Alt+W";
    public DateTime TimestampUtc { get; set; } = DateTime.UtcNow;
    public List<WindowSnapshot> Windows { get; set; } = new();
}

internal sealed class WindowSnapshot
{
    public string Hwnd { get; set; } = string.Empty;
    public string Title { get; set; } = string.Empty;
    public int ProcessId { get; set; }
    public RectInt Rect { get; set; } = new();
    public DateTime TimestampUtc { get; set; }
    public UiaNode? Uia { get; set; }
    public string? UiaError { get; set; }
}

internal sealed class RectInt
{
    public int Left { get; set; }
    public int Top { get; set; }
    public int Right { get; set; }
    public int Bottom { get; set; }
}

internal sealed class RectDouble
{
    public double Left { get; set; }
    public double Top { get; set; }
    public double Right { get; set; }
    public double Bottom { get; set; }
}

internal sealed class UiaNode
{
    public string? Name { get; set; }
    public string? AutomationId { get; set; }
    public string? ControlType { get; set; }
    public string? ClassName { get; set; }
    public string? FrameworkId { get; set; }
    public RectDouble? BoundingRect { get; set; }
    public bool? IsOffscreen { get; set; }
    public int? ProcessId { get; set; }
    public List<UiaNode> Children { get; set; } = new();
}
