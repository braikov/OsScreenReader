using System;

namespace WindowUiaSnapshot.Models;

/// <summary>
/// Describes a top-level window and its UI Automation snapshot.
/// </summary>
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
