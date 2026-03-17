# ============================================================
# AEDA Framework - Script de Configuración del Entorno
# Uso: .\setup_env.ps1
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

# ---- 1. Verificar Python ----
Write-Header "Verificando Python"
try {
    $pyVersion = python --version 2>&1
    Write-OK "Encontrado: $pyVersion"
} catch {
    Write-ERR "Python no encontrado. Instala Python 3.11+ desde https://python.org"
    exit 1
}

# ---- 2. Configuración del entorno virtual ----
Write-Header "Configurando Entorno Virtual"

if ($UseConda) {
    Write-Host "Usando Conda..." -ForegroundColor Yellow
    try {
        conda env create -f "$baseDir\environment.yml" --force
        Write-OK "Entorno conda 'aeda-framework' creado."
        Write-Host "`nActiva el entorno con: conda activate aeda-framework" -ForegroundColor Yellow
    } catch {
        Write-ERR "Fallo al crear entorno conda: $_"
        exit 1
    }
} else {
    $venvPath = Join-Path $baseDir ".venv"
    if (-not (Test-Path $venvPath)) {
        Write-Host "Creando entorno virtual en .venv..." -ForegroundColor Yellow
        python -m venv $venvPath
        Write-OK "Entorno virtual creado en: $venvPath"
    } else {
        Write-OK "Entorno virtual ya existe en: $venvPath"
    }

    # Activar entorno virtual
    $activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
    if (Test-Path $activateScript) {
        & $activateScript
        Write-OK "Entorno virtual activado."
    } else {
        Write-ERR "No se encontró el script de activación en $activateScript"
        exit 1
    }

    # ---- 3. Instalar dependencias ----
    if (-not $SkipInstall) {
        Write-Header "Instalando Dependencias"
        Write-Host "Actualizando pip..." -ForegroundColor Yellow
        python -m pip install --upgrade pip --quiet

        Write-Host "Instalando paquetes desde requirements.txt..." -ForegroundColor Yellow
        pip install -r "$baseDir\requirements.txt"
        Write-OK "Dependencias instaladas exitosamente."
    }
}

# ---- 4. Verificación del entorno ----
if ($Verify -or (-not $SkipInstall)) {
    Write-Header "Verificando Instalación"

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
                Write-WARN ("{0,-15} -> NO INSTALADO" -f $check.name)
                $failed++
            }
        } catch {
            Write-WARN ("{0,-15} -> ERROR: {1}" -f $check.name, $_)
            $failed++
        }
    }

    Write-Header "Resumen"
    Write-Host "  Paquetes verificados: $($passed + $failed)" -ForegroundColor White
    Write-OK   "  Instalados correctamente: $passed"
    if ($failed -gt 0) {
        Write-WARN "  Fallos: $failed (ejecuta 'pip install -r requirements.txt' manualmente)"
    } else {
        Write-Host "`n  El entorno AEDA esta listo." -ForegroundColor Green
    }
}

Write-Header "Listo"
Write-Host "Siguiente paso: jupyter lab" -ForegroundColor Cyan
