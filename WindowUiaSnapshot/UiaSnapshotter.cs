using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Automation;
using WindowUiaSnapshot.Models;

namespace WindowUiaSnapshot;

/// <summary>
/// Builds UI Automation trees with depth/child limits and timeouts.
/// </summary>
internal sealed class UiaSnapshotter
{
    private readonly int _maxDepth;
    private readonly int _maxChildrenPerNode;
    private readonly int _timeoutPerWindowMs;

    public UiaSnapshotter(int maxDepth = 8, int maxChildrenPerNode = 200, int timeoutPerWindowMs = 1500)
    {
        _maxDepth = maxDepth;
        _maxChildrenPerNode = maxChildrenPerNode;
        _timeoutPerWindowMs = timeoutPerWindowMs;
    }

    /// <summary>
    /// Capture a UIA subtree for a given window handle with timeout.
    /// </summary>
    public async Task<(UiaNode? node, string? error)> SnapshotAsync(IntPtr hwnd)
    {
        using var cts = new CancellationTokenSource(_timeoutPerWindowMs);
        try
        {
            var result = await Task.Run(() => BuildTree(hwnd, cts.Token), cts.Token).ConfigureAwait(false);
            return (result, null);
        }
        catch (OperationCanceledException)
        {
            return (null, "timeout");
        }
        catch (Exception ex)
        {
            return (null, ex.Message);
        }
    }

    private UiaNode? BuildTree(IntPtr hwnd, CancellationToken token)
    {
        token.ThrowIfCancellationRequested();
        AutomationElement element;
        try
        {
            element = AutomationElement.FromHandle(hwnd);
        }
        catch (Exception ex) when (IsExpected(ex))
        {
            return null;
        }

        return BuildNode(element, 0, token);
    }

    private UiaNode? BuildNode(AutomationElement element, int depth, CancellationToken token)
    {
        token.ThrowIfCancellationRequested();
        if (element == null)
        {
            return null;
        }

        try
        {
            var current = element.Current;
            var node = new UiaNode
            {
                Name = SafeString(current.Name),
                AutomationId = SafeString(current.AutomationId),
                ControlType = current.ControlType?.ProgrammaticName,
                ClassName = SafeString(current.ClassName),
                FrameworkId = SafeString(current.FrameworkId),
                BoundingRect = current.BoundingRectangle.IsEmpty
                    ? null
                    : new RectDouble
                    {
                        Left = current.BoundingRectangle.Left,
                        Top = current.BoundingRectangle.Top,
                        Right = current.BoundingRectangle.Right,
                        Bottom = current.BoundingRectangle.Bottom,
                    },
                IsOffscreen = current.IsOffscreen,
                ProcessId = current.ProcessId,
            };

            if (depth >= _maxDepth)
            {
                return node;
            }

            var children = GetChildren(element, token);
            var count = Math.Min(children.Count, _maxChildrenPerNode);
            for (var i = 0; i < count; i++)
            {
                var child = children[i];
                var childNode = BuildNode(child, depth + 1, token);
                if (childNode != null)
                {
                    node.Children.Add(childNode);
                }
            }

            return node;
        }
        catch (Exception ex) when (IsExpected(ex))
        {
            return null;
        }
    }

    private static List<AutomationElement> GetChildren(AutomationElement parent, CancellationToken token)
    {
        token.ThrowIfCancellationRequested();
        var list = new List<AutomationElement>();
        try
        {
            var walker = TreeWalker.RawViewWalker;
            var child = walker.GetFirstChild(parent);
            while (child != null)
            {
                list.Add(child);
                child = walker.GetNextSibling(child);
                token.ThrowIfCancellationRequested();
            }
        }
        catch (Exception ex) when (IsExpected(ex))
        {
            // ignore and return what we have
        }

        return list;
    }

    private static string? SafeString(string? value) => string.IsNullOrWhiteSpace(value) ? null : value;

    private static bool IsExpected(Exception ex) =>
        ex is ElementNotAvailableException
        || ex is COMException
        || ex is UnauthorizedAccessException;
}
