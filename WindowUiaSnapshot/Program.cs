using System;
using System.Windows.Forms;

namespace WindowUiaSnapshot;

/// <summary>
/// Application entry point wiring the tray context.
/// </summary>
internal static class Program
{
    [STAThread]
    private static void Main()
    {
        ApplicationConfiguration.Initialize();
        Application.Run(new TrayAppContext());
    }
}
