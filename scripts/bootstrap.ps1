# One-shot environment bootstrap (Windows PowerShell 5.1+).
#
#   .\scripts\bootstrap.ps1
#
# Requires: uv (https://docs.astral.sh/uv/) in PATH.
# Creates .venv from requirements.lock.txt and verifies the vendored assets
# and the model+policy load before printing the reproduction command map.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

function Assert-Cmd($name) {
  if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
    Write-Host "[fail] '$name' not found in PATH. Install it and retry." -ForegroundColor Red
    exit 1
  }
}

Assert-Cmd "uv"

Write-Host "[bootstrap] creating .venv (Python 3.12)..." -ForegroundColor Cyan
uv venv (Join-Path $Root ".venv") --python 3.12

Write-Host "[bootstrap] installing locked requirements..." -ForegroundColor Cyan
uv pip install --python (Join-Path $Root ".venv\Scripts\python.exe") -r (Join-Path $Root "requirements.lock.txt")

$Py = Join-Path $Root ".venv\Scripts\python.exe"

Write-Host "[bootstrap] verifying vendored assets..." -ForegroundColor Cyan
$motion = Join-Path $Root "third_party\unitree_rl_gym\deploy\pre_train\g1\motion.pt"
$meshes = Join-Path $Root "third_party\unitree_rl_gym\resources\robots\g1_description\meshes"
if (-not (Test-Path $motion)) { Write-Host "[fail] missing $motion" -ForegroundColor Red; exit 1 }
if (-not (Test-Path $meshes)) { Write-Host "[fail] missing $meshes" -ForegroundColor Red; exit 1 }

Write-Host "[bootstrap] smoke-testing model + policy load..." -ForegroundColor Cyan
& $Py -c "import sys; sys.path.insert(0, r'$Root\deploy'); import mujoco; from deploy12 import Deploy12, RLGYM; m=mujoco.MjModel.from_xml_path(str(RLGYM/'../../data/g1_description/scene_curb.xml')); d=Deploy12(); d.reset(cmd=[0.5,0,0]); print('scene nq=%d, policy loaded, ok' % m.nq)"

Write-Host "[bootstrap] done." -ForegroundColor Green
Write-Host ""
Write-Host "Next:"
Write-Host "  .\.venv\Scripts\activate"
Write-Host "  python src\task_tracking.py                 # primary battery (5x5x6 seeds)"
Write-Host "  python src\task_tracking.py --substrate izh # substrate interchangeability"
Write-Host "  python paper\figures\build_figures.py       # paper figures from JSONs"
Write-Host "See docs\REPRODUCE.md for the full command map."