using System;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;
using WindowUiaSnapshot.Models;

namespace WindowUiaSnapshot;

/// <summary>
/// Serializes snapshots to disk under the local app data folder.
/// </summary>
internal sealed class SnapshotWriter
{
    private readonly string _outputRoot;

    public SnapshotWriter()
    {
        var localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        _outputRoot = Path.Combine(localAppData, "WindowUiaSnapshot", "snapshots");
    }

    public string EnsureOutputDirectory()
    {
        Directory.CreateDirectory(_outputRoot);
        return _outputRoot;
    }

    public async Task<string> WriteAsync(SnapshotRoot root)
    {
        EnsureOutputDirectory();
        var fileName = $"snapshot_{DateTime.UtcNow:yyyyMMdd_HHmmss}.json";
        var path = Path.Combine(_outputRoot, fileName);

        var options = new JsonSerializerOptions
        {
            WriteIndented = true,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        };

        await File.WriteAllTextAsync(path, JsonSerializer.Serialize(root, options)).ConfigureAwait(false);
        return path;
    }
}
