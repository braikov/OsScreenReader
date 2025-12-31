using System;
using System.Collections.Generic;

namespace WindowUiaSnapshot.Models;

/// <summary>
/// Root snapshot metadata and collection of window snapshots.
/// </summary>
internal sealed class SnapshotRoot
{
    public string MachineName { get; set; } = Environment.MachineName;
    public string UserName { get; set; } = Environment.UserName;
    public string OsVersion { get; set; } = Environment.OSVersion.ToString();
    public string Hotkey { get; set; } = "Ctrl+Alt+W";
    public DateTime TimestampUtc { get; set; } = DateTime.UtcNow;
    public List<WindowSnapshot> Windows { get; set; } = new();
}
