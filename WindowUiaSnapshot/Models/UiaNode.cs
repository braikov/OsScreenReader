using System.Collections.Generic;

namespace WindowUiaSnapshot.Models;

/// <summary>
/// UI Automation node captured from the visual tree.
/// </summary>
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
