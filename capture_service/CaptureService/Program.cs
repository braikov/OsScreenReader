using System.Configuration;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Windows.Forms;

namespace CaptureService;

internal static class Program
{
    /// <summary>
    /// Application entry point for the capture service.
    /// </summary>
    [STAThread]
    private static void Main()
    {
        Application.SetHighDpiMode(HighDpiMode.SystemAware);
        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);

        using var context = new HotkeyContext();
        Application.Run(context);
    }
}

/// <summary>
/// Application context that owns the hotkey window and capture lifecycle.
/// </summary>
internal sealed class HotkeyContext : ApplicationContext
{
    private readonly HotkeyWindow _window;

    /// <summary>
    /// Initializes the hotkey context and registers the capture handler.
    /// </summary>
    public HotkeyContext()
    {
        _window = new HotkeyWindow();
        _window.HotkeyPressed += (_, _) => CaptureScreenshot();
    }

    /// <summary>
    /// Releases resources when the application context is disposed.
    /// </summary>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _window.Dispose();
        }

        base.Dispose(disposing);
    }

    /// <summary>
    /// Captures the baseline and grid frame screenshots for the active session.
    /// </summary>
    private static void CaptureScreenshot()
    {
        var rootFolder = ConfigurationManager.AppSettings["RootFolder"];
        if (string.IsNullOrWhiteSpace(rootFolder))
        {
            MessageBox.Show(
                "RootFolder is not configured in app.config.",
                "Capture Service",
                MessageBoxButtons.OK,
                MessageBoxIcon.Error);
            return;
        }

        var stepPixels = ReadPositiveIntSetting("StepPixels", 25);
        var maxScreenshots = ReadPositiveIntSetting("MaxScreenshots", 2000);
        var interCaptureDelayMs = ReadPositiveIntSetting("InterCaptureDelayMs", 2000);

        Directory.CreateDirectory(rootFolder);
        var sessionId = DateTimeOffset.UtcNow.ToString("yyyyMMdd_HHmmss");
        var sessionFolder = Path.Combine(rootFolder, sessionId);
        Directory.CreateDirectory(sessionFolder);

        var bounds = Screen.PrimaryScreen.Bounds;
        var baselineBytes = CaptureScreenshot(bounds, out var baselineSize);
        var baselineHash = ComputeHash(baselineBytes);
        SaveScreenshot(Path.Combine(sessionFolder, "baseline.png"), baselineBytes);

        var framesFolder = Path.Combine(sessionFolder, "frames");
        Directory.CreateDirectory(framesFolder);

        var savedCount = 0;
        var attemptedCount = 0;
        var previousHash = baselineHash;
        for (var y = bounds.Top + stepPixels; y < bounds.Bottom; y += stepPixels)
        {
            for (var x = bounds.Left; x < bounds.Right; x += stepPixels)
            {
                attemptedCount++;
                if (attemptedCount > maxScreenshots)
                {
                    MessageBox.Show(
                        $"Reached max screenshots limit ({maxScreenshots}) after {savedCount} saved.",
                        "Capture Service",
                        MessageBoxButtons.OK,
                        MessageBoxIcon.Information);
                    return;
                }

                Cursor.Position = new Point(x, y);
                Thread.Sleep(interCaptureDelayMs);
                var framePath = Path.Combine(framesFolder, $"frame_{savedCount + 1:D6}_{x}_{y}.png");
                if (TrySaveScreenshotIfDifferent(framePath, bounds, baselineHash, previousHash, out var currentHash, out var newSize))
                {
                    previousHash = currentHash;
                    savedCount++;
                }
            }
        }

        MessageBox.Show(
            $"Saved baseline and {savedCount} frames (attempted {attemptedCount}) to {sessionFolder}",
            "Capture Service",
            MessageBoxButtons.OK,
            MessageBoxIcon.Information);
    }

    /// <summary>
    /// Captures the screen bounds and returns the PNG bytes.
    /// </summary>
    private static byte[] CaptureScreenshot(Rectangle bounds, out long fileSize)
    {
        using var bitmap = new Bitmap(bounds.Width, bounds.Height);
        using (var graphics = Graphics.FromImage(bitmap))
        {
            graphics.CopyFromScreen(bounds.Left, bounds.Top, 0, 0, bounds.Size);
        }

        using var memoryStream = new MemoryStream();
        bitmap.Save(memoryStream, System.Drawing.Imaging.ImageFormat.Png);
        fileSize = memoryStream.Length;
        return memoryStream.ToArray();
    }

    /// <summary>
    /// Saves a PNG byte array to disk.
    /// </summary>
    private static void SaveScreenshot(string path, byte[] pngBytes)
    {
        File.WriteAllBytes(path, pngBytes);
    }

    /// <summary>
    /// Captures the screen and saves only when the content differs from the baseline and previous capture.
    /// </summary>
    private static bool TrySaveScreenshotIfDifferent(
        string path,
        Rectangle bounds,
        byte[] baselineHash,
        byte[] previousHash,
        out byte[] currentHash,
        out long fileSize)
    {
        var pngBytes = CaptureScreenshot(bounds, out fileSize);
        if (fileSize == 0)
        {
            currentHash = Array.Empty<byte>();
            return false;
        }

        currentHash = ComputeHash(pngBytes);
        if (HashesEqual(currentHash, previousHash) || HashesEqual(currentHash, baselineHash))
        {
            return false;
        }

        SaveScreenshot(path, pngBytes);
        return true;
    }

    private static byte[] ComputeHash(byte[] data)
    {
        using var sha = SHA256.Create();
        return sha.ComputeHash(data);
    }

    private static bool HashesEqual(byte[] a, byte[] b)
    {
        if (a.Length != b.Length)
        {
            return false;
        }

        for (var i = 0; i < a.Length; i++)
        {
            if (a[i] != b[i])
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>
    /// Reads a positive integer setting with a fallback default.
    /// </summary>
    private static int ReadPositiveIntSetting(string key, int defaultValue)
    {
        var rawValue = ConfigurationManager.AppSettings[key];
        return int.TryParse(rawValue, out var parsed) && parsed > 0 ? parsed : defaultValue;
    }
}

/// <summary>
/// Hidden native window that registers and listens for the hotkey.
/// </summary>
internal sealed class HotkeyWindow : NativeWindow, IDisposable
{
    private const int HotkeyId = 1;
    private const int WmHotkey = 0x0312;
    private readonly IntPtr _hWnd;

    public event EventHandler? HotkeyPressed;

    /// <summary>
    /// Creates an invisible window and registers the Ctrl+PrintScreen hotkey.
    /// </summary>
    public HotkeyWindow()
    {
        var cp = new CreateParams();
        CreateHandle(cp);
        _hWnd = Handle;

        if (!RegisterHotKey(_hWnd, HotkeyId, MOD_CONTROL, (int)Keys.PrintScreen))
        {
            throw new InvalidOperationException("Failed to register hotkey Ctrl+PrintScreen.");
        }
    }

    /// <summary>
    /// Handles hotkey messages and notifies subscribers.
    /// </summary>
    protected override void WndProc(ref Message m)
    {
        if (m.Msg == WmHotkey && m.WParam.ToInt32() == HotkeyId)
        {
            HotkeyPressed?.Invoke(this, EventArgs.Empty);
        }

        base.WndProc(ref m);
    }

    /// <summary>
    /// Unregisters the hotkey and releases the native window handle.
    /// </summary>
    public void Dispose()
    {
        UnregisterHotKey(_hWnd, HotkeyId);
        DestroyHandle();
    }

    private const int MOD_CONTROL = 0x0002;

    [DllImport("user32.dll", SetLastError = true)]
    private static extern bool RegisterHotKey(IntPtr hWnd, int id, int fsModifiers, int vk);

    [DllImport("user32.dll", SetLastError = true)]
    private static extern bool UnregisterHotKey(IntPtr hWnd, int id);
}
