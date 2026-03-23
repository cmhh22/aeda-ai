# ============================================================
# AEDA Framework - Environment Setup Script
# Usage: .\setup_env.ps1
# ============================================================

param(
    [switch]$UseConda,
    [switch]$SkipInstall,
    [switch]$Verify
)

$ErrorActionPreference = "Stop"
$baseDir = $PSScriptRoot

function Write-Header($text) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "  $text" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
}

function Write-OK($text)   { Write-Host "[OK] $text" -ForegroundColor Green }
function Write-WARN($text) { Write-Host "[WARN] $text" -ForegroundColor Yellow }
function Write-ERR($text)  { Write-Host "[ERROR] $text" -ForegroundColor Red }

# ---- 1. Verify Python ----
Write-Header "Verifying Python"
try {
    $pyVersion = python --version 2>&1
    Write-OK "Found: $pyVersion"
} catch {
    Write-ERR "Python not found. Install Python 3.11+ from https://python.org"
    exit 1
}

# ---- 2. Virtual environment setup ----
Write-Header "Configuring Virtual Environment"

if ($UseConda) {
    Write-Host "Using Conda..." -ForegroundColor Yellow
    try {
        conda env create -f "$baseDir\environment.yml" --force
        Write-OK "Conda environment 'aeda-framework' created."
        Write-Host "`nActivate it with: conda activate aeda-framework" -ForegroundColor Yellow
    } catch {
        Write-ERR "Failed to create conda environment: $_"
        exit 1
    }
} else {
    $venvPath = Join-Path $baseDir ".venv"
    if (-not (Test-Path $venvPath)) {
        Write-Host "Creating virtual environment in .venv..." -ForegroundColor Yellow
        python -m venv $venvPath
        Write-OK "Virtual environment created at: $venvPath"
    } else {
        Write-OK "Virtual environment already exists at: $venvPath"
    }

    # Activate virtual environment
    $activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
    if (Test-Path $activateScript) {
        & $activateScript
        Write-OK "Virtual environment activated."
    } else {
        Write-ERR "Activation script not found at $activateScript"
        exit 1
    }

    # ---- 3. Install dependencies ----
    if (-not $SkipInstall) {
        Write-Header "Installing Dependencies"
        Write-Host "Upgrading pip..." -ForegroundColor Yellow
        python -m pip install --upgrade pip --quiet

        Write-Host "Installing packages from requirements.txt..." -ForegroundColor Yellow
        pip install -r "$baseDir\requirements.txt"
        Write-OK "Dependencies installed successfully."
    }
}

# ---- 4. Environment verification ----
if ($Verify -or (-not $SkipInstall)) {
    Write-Header "Verifying Installation"

    $checks = @(
        @{ name = "numpy";       import = "import numpy as np; print(np.__version__)" },
        @{ name = "pandas";      import = "import pandas as pd; print(pd.__version__)" },
        @{ name = "scipy";       import = "import scipy; print(scipy.__version__)" },
        @{ name = "sklearn";     import = "import sklearn; print(sklearn.__version__)" },
        @{ name = "pyod";        import = "import pyod; print(pyod.__version__)" },
        @{ name = "missingno";   import = "import missingno; print('OK')" },
        @{ name = "miceforest";  import = "import miceforest; print(miceforest.__version__)" },
        @{ name = "umap-learn";  import = "import umap; print(umap.__version__)" },
        @{ name = "shap";        import = "import shap; print(shap.__version__)" },
        @{ name = "lime";        import = "import lime; print('OK')" },
        @{ name = "matplotlib";  import = "import matplotlib; print(matplotlib.__version__)" },
        @{ name = "seaborn";     import = "import seaborn as sns; print(sns.__version__)" },
        @{ name = "plotly";      import = "import plotly; print(plotly.__version__)" }
    )

    $passed = 0
    $failed = 0

    foreach ($check in $checks) {
        try {
            $result = python -c $check.import 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-OK ("{0,-15} -> v{1}" -f $check.name, $result)
                $passed++
            } else {
                Write-WARN ("{0,-15} -> NOT INSTALLED" -f $check.name)
                $failed++
            }
        } catch {
            Write-WARN ("{0,-15} -> ERROR: {1}" -f $check.name, $_)
            $failed++
        }
    }

    Write-Header "Summary"
    Write-Host "  Packages checked: $($passed + $failed)" -ForegroundColor White
    Write-OK   "  Installed successfully: $passed"
    if ($failed -gt 0) {
        Write-WARN "  Failures: $failed (run 'pip install -r requirements.txt' manually)"
    } else {
        Write-Host "`n  The AEDA environment is ready." -ForegroundColor Green
    }
}

Write-Header "Done"
Write-Host "Next step: jupyter lab" -ForegroundColor Cyan
