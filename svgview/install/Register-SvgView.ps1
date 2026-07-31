<#
.SYNOPSIS
    Registers svgview.exe as a handler for .svg / .svgz files for the current user.

.DESCRIPTION
    Writes a ProgID under HKCU:\Software\Classes and adds it to the
    OpenWithProgids list for .svg and .svgz. Everything is per-user, so no
    administrator rights are needed and nothing outside HKCU is touched.

    This deliberately does NOT seize the default handler. Since Windows 8 the
    default association can only be changed by the user through the Settings
    or "Open with" UI -- any program that claims to do it silently is either
    lying or breaking. After running this, right-click an .svg file, choose
    "Open with" > "Choose another app" > svgview > "Always use this app".

.PARAMETER ExePath
    Path to svgview.exe. Defaults to ..\target\release\svgview.exe relative to
    this script.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\Register-SvgView.ps1
#>
[CmdletBinding()]
param(
    [string]$ExePath
)

$ErrorActionPreference = 'Stop'

if (-not $ExePath) {
    $ExePath = Join-Path $PSScriptRoot '..\target\release\svgview.exe'
}
$ExePath = (Resolve-Path -LiteralPath $ExePath).Path

if (-not (Test-Path -LiteralPath $ExePath -PathType Leaf)) {
    throw "svgview.exe not found at $ExePath. Build it first: cargo build --release"
}

$progId = 'svgview.svgfile'
$classes = 'HKCU:\Software\Classes'

function Set-Value($Path, $Name, $Value) {
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -Path $Path -Force | Out-Null
    }
    New-ItemProperty -LiteralPath $Path -Name $Name -Value $Value -PropertyType String -Force | Out-Null
}

# The ProgID itself: friendly name, icon, and the open verb.
Set-Value "$classes\$progId"                    '(default)' 'SVG Image'
Set-Value "$classes\$progId\DefaultIcon"        '(default)' "`"$ExePath`",0"
Set-Value "$classes\$progId\shell\open"         'FriendlyAppName' 'svgview'
Set-Value "$classes\$progId\shell\open\command" '(default)' "`"$ExePath`" `"%1`""

# Offer svgview in the "Open with" list for both extensions, without
# displacing whatever handler is currently the default.
foreach ($ext in '.svg', '.svgz') {
    $key = "$classes\$ext\OpenWithProgids"
    if (-not (Test-Path -LiteralPath $key)) {
        New-Item -Path $key -Force | Out-Null
    }
    # An empty REG_NONE value is the documented shape for this list.
    New-ItemProperty -LiteralPath $key -Name $progId -Value ([byte[]]@()) `
        -PropertyType None -Force | Out-Null
}

# Registering the application makes it show up in "Choose another app".
Set-Value "$classes\Applications\svgview.exe\shell\open\command" '(default)' "`"$ExePath`" `"%1`""
$supported = "$classes\Applications\svgview.exe\SupportedTypes"
foreach ($ext in '.svg', '.svgz') {
    Set-Value $supported $ext ''
}

# Tell Explorer to re-read associations (SHCNE_ASSOCCHANGED).
$signature = @'
[DllImport("shell32.dll", CharSet = CharSet.Unicode)]
public static extern void SHChangeNotify(int eventId, uint flags, IntPtr item1, IntPtr item2);
'@
try {
    $shell = Add-Type -MemberDefinition $signature -Name 'SvgViewShell' -Namespace 'Win32' -PassThru
    $shell::SHChangeNotify(0x08000000, 0x0000, [IntPtr]::Zero, [IntPtr]::Zero)
} catch {
    Write-Warning "Could not notify Explorer; sign out and back in if the icon looks stale. $_"
}

Write-Host "Registered $progId -> $ExePath"
Write-Host 'To make it the default: right-click an .svg file > Open with > Choose another app > svgview > Always.'
