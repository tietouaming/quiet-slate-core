param(
    [string]$VenvPath = ".venv_sim",
    [string]$PythonExe = "python",
    [ValidateSet("auto", "cpu", "cu126", "cu124", "cu121")]
    [string]$TorchVariant = "auto",
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"

if ($Recreate -and (Test-Path $VenvPath)) {
    Write-Host "Removing existing environment: $VenvPath"
    Remove-Item -Recurse -Force $VenvPath
}

$PyVenv = Join-Path $VenvPath "Scripts\\python.exe"
if (-not (Test-Path $PyVenv)) {
    Write-Host "Creating virtual environment: $VenvPath"
    & $PythonExe -m venv $VenvPath
}

function Invoke-Pip {
    param([string[]]$PipArgs)
    & $PyVenv -m pip @PipArgs
    if ($LASTEXITCODE -ne 0) {
        throw "pip failed: $($PipArgs -join ' ')"
    }
}

function Install-Torch {
    param([string]$Variant)
    if ($Variant -eq "cpu") {
        Invoke-Pip -PipArgs @("install", "--upgrade", "torch", "--index-url", "https://download.pytorch.org/whl/cpu")
        return
    }
    Invoke-Pip -PipArgs @("install", "--upgrade", "torch", "--index-url", "https://download.pytorch.org/whl/$Variant")
}

function Detect-TorchVariant {
    param([string]$Requested)
    if ($Requested -ne "auto") {
        return $Requested
    }
    $hasNvidia = $null -ne (Get-Command nvidia-smi -ErrorAction SilentlyContinue)
    if ($hasNvidia) {
        return "cu126"
    }
    return "cpu"
}

Write-Host "Installing dependencies..."
Invoke-Pip -PipArgs @("install", "--upgrade", "pip", "setuptools", "wheel")

$tmpReq = Join-Path $env:TEMP "requirements_sim_no_torch_$PID.txt"
Get-Content "requirements_sim.txt" |
    Where-Object {
        $_.Trim() -ne "" -and
        -not $_.Trim().StartsWith("#") -and
        $_ -notmatch "^\s*torch([<>=!].*)?$"
    } |
    Set-Content $tmpReq
Invoke-Pip -PipArgs @("install", "-r", $tmpReq)

$targetVariant = Detect-TorchVariant -Requested $TorchVariant
if ($targetVariant -eq "cpu") {
    Install-Torch -Variant "cpu"
}
else {
    $attempts = @($targetVariant)
    if ($TorchVariant -eq "auto") {
        foreach ($v in @("cu124", "cu121")) {
            if ($attempts -notcontains $v) {
                $attempts += $v
            }
        }
    }
    $installed = $false
    foreach ($v in $attempts) {
        try {
            Write-Host "Trying PyTorch variant: $v"
            Install-Torch -Variant $v
            $installed = $true
            break
        }
        catch {
            Write-Host "PyTorch $v install failed, trying next fallback..."
        }
    }
    if (-not $installed) {
        if ($TorchVariant -eq "auto") {
            Write-Host "Falling back to CPU torch."
            Install-Torch -Variant "cpu"
        }
        else {
            throw "Unable to install requested torch variant: $TorchVariant"
        }
    }
}
Remove-Item $tmpReq -ErrorAction SilentlyContinue

Write-Host "Environment summary:"
& $PyVenv -c "import sys, torch; print('python', sys.version.split()[0]); print('torch', torch.__version__); print('torch_cuda', torch.version.cuda); print('cuda_available', torch.cuda.is_available()); print('cuda_device_count', torch.cuda.device_count())"

Write-Host "Done. Use: $PyVenv scripts/run_notched_case.py --config configs/notch_case.yaml --device cuda --disable-ml"
