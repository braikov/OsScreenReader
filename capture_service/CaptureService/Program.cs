using System.Configuration;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Windows.Forms;

namespace CaptureService;

internal static class Program
{
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

internal sealed class HotkeyContext : ApplicationContext
{
    private readonly HotkeyWindow _window;

    public HotkeyContext()
    {
        _window = new HotkeyWindow();
        _window.HotkeyPressed += (_, _) => CaptureScreenshot();
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _window.Dispose();
        }

        base.Dispose(disposing);
    }

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

        Directory.CreateDirectory(rootFolder);
        var sessionId = DateTimeOffset.UtcNow.ToString("yyyyMMdd_HHmmss");
        var sessionFolder = Path.Combine(rootFolder, sessionId);
        Directory.CreateDirectory(sessionFolder);

        var bounds = Screen.PrimaryScreen.Bounds;
        using var bitmap = new Bitmap(bounds.Width, bounds.Height);
        using (var graphics = Graphics.FromImage(bitmap))
        {
            graphics.CopyFromScreen(bounds.Left, bounds.Top, 0, 0, bounds.Size);
        }

        var outputPath = Path.Combine(sessionFolder, "baseline.png");
        bitmap.Save(outputPath, System.Drawing.Imaging.ImageFormat.Png);

        MessageBox.Show(
            $"Saved screenshot to {outputPath}",
            "Capture Service",
            MessageBoxButtons.OK,
            MessageBoxIcon.Information);
    }
}

internal sealed class HotkeyWindow : NativeWindow, IDisposable
{
    private const int HotkeyId = 1;
    private const int WmHotkey = 0x0312;
    private readonly IntPtr _hWnd;

    public event EventHandler? HotkeyPressed;

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

    protected override void WndProc(ref Message m)
    {
        if (m.Msg == WmHotkey && m.WParam.ToInt32() == HotkeyId)
        {
            HotkeyPressed?.Invoke(this, EventArgs.Empty);
        }

        base.WndProc(ref m);
    }

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
