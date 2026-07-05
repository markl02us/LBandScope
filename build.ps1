# Build BOTH artifacts and gate on the built-in self-test:
#   LBandScope.exe   - double-click GUI (what beginners use), windowed
#   lbandscope.exe   - command-line tool (power users / scripting), console
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "==> Building GUI (LBandScope.exe)" -ForegroundColor Cyan
python -m PyInstaller --onefile --windowed --name LBandScope `
    --distpath dist --workpath build --specpath build `
    --clean --noconfirm --hidden-import numpy entry_gui.py | Out-Null

# NOTE: name must differ from LBandScope by more than case -- Windows filenames
# are case-insensitive, so "lbandscope" would overwrite "LBandScope".
Write-Host "==> Building CLI (lbandscope-cli.exe)" -ForegroundColor Cyan
python -m PyInstaller --onefile --name lbandscope-cli `
    --distpath dist --workpath build --specpath build `
    --clean --noconfirm --hidden-import numpy entry.py | Out-Null

$gui = Join-Path $PSScriptRoot "dist\LBandScope.exe"
$cli = Join-Path $PSScriptRoot "dist\lbandscope-cli.exe"
foreach ($e in @($gui, $cli)) {
    if (-not (Test-Path $e)) { throw "build failed: $e missing" }
    $mb = "{0:N1}" -f ((Get-Item $e).Length / 1MB)
    Write-Host "==> Built $e ($mb MB)" -ForegroundColor Green
}

Write-Host "==> Acceptance: running CLI self-test" -ForegroundColor Cyan
& $cli selftest
if ($LASTEXITCODE -ne 0) { throw "SELFTEST FAILED on built binary" }
Write-Host "==> RELEASE OK  (give away LBandScope.exe -- users just double-click it)" `
    -ForegroundColor Green
