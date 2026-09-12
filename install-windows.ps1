$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
py -3.12 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.12 first.' }
& .\.venv\Scripts\python.exe -m pip install -r requirements-semantic.txt
if ($LASTEXITCODE -ne 0) { throw 'Python dependency installation failed.' }
corepack pnpm install
if ($LASTEXITCODE -ne 0) { throw 'Node dependencies failed. Install Node 22 LTS and Corepack.' }
corepack pnpm build
$env:SPLAT_PYTHON = "$PSScriptRoot\.venv\Scripts\python.exe"
corepack pnpm desktop
