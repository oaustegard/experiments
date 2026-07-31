<#
.SYNOPSIS
    Removes the per-user svgview file associations written by Register-SvgView.ps1.

.DESCRIPTION
    Deletes the ProgID, the Applications entry, and the OpenWithProgids
    references. Only touches HKCU, and leaves the .svg / .svgz keys themselves
    alone -- other applications register there too.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\Unregister-SvgView.ps1
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

$progId = 'svgview.svgfile'
$classes = 'HKCU:\Software\Classes'

foreach ($key in "$classes\$progId", "$classes\Applications\svgview.exe") {
    if (Test-Path -LiteralPath $key) {
        Remove-Item -LiteralPath $key -Recurse -Force
        Write-Host "Removed $key"
    }
}

foreach ($ext in '.svg', '.svgz') {
    $key = "$classes\$ext\OpenWithProgids"
    if (Test-Path -LiteralPath $key) {
        # -ErrorAction SilentlyContinue: the value is absent if registration
        # never ran for this extension, which is not an error here.
        Remove-ItemProperty -LiteralPath $key -Name $progId -Force -ErrorAction SilentlyContinue
        Write-Host "Removed $progId from $key"
    }
}

# If svgview was the chosen default, Windows keeps a UserChoice entry that we
# are not permitted to delete. Point the user at the UI instead of failing.
$userChoice = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Explorer\FileExts\.svg\UserChoice"
if (Test-Path -LiteralPath $userChoice) {
    $current = (Get-ItemProperty -LiteralPath $userChoice -ErrorAction SilentlyContinue).ProgId
    if ($current -eq $progId) {
        Write-Warning 'svgview is still the default for .svg. Change it in Settings > Apps > Default apps.'
    }
}

$signature = @'
[DllImport("shell32.dll", CharSet = CharSet.Unicode)]
public static extern void SHChangeNotify(int eventId, uint flags, IntPtr item1, IntPtr item2);
'@
try {
    $shell = Add-Type -MemberDefinition $signature -Name 'SvgViewShellRemove' -Namespace 'Win32' -PassThru
    $shell::SHChangeNotify(0x08000000, 0x0000, [IntPtr]::Zero, [IntPtr]::Zero)
} catch {
    Write-Warning "Could not notify Explorer. $_"
}

Write-Host 'Done.'
