param(
    [string]$PythonExe
)

$ErrorActionPreference = "Stop"
$projectDir = $PSScriptRoot

function Resolve-PythonExecutable {
    param(
        [string]$Override
    )

    if (-not [string]::IsNullOrWhiteSpace($Override)) {
        if (Test-Path -LiteralPath $Override -PathType Leaf) {
            return (Resolve-Path -LiteralPath $Override).Path
        }

        $overrideCommand = Get-Command $Override -CommandType Application -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($null -ne $overrideCommand) {
            return $overrideCommand.Source
        }

        throw "Python interpreter not found: $Override"
    }

    if (-not [string]::IsNullOrWhiteSpace($env:CONDA_PREFIX)) {
        $condaPython = Join-Path $env:CONDA_PREFIX "python.exe"
        if (Test-Path -LiteralPath $condaPython -PathType Leaf) {
            return (Resolve-Path -LiteralPath $condaPython).Path
        }
    }

    $pathPython = Get-Command python -CommandType Application -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($null -ne $pathPython) {
        return $pathPython.Source
    }

    throw "Python interpreter not found. Activate the Conda environment, add Python to PATH, or pass -PythonExe explicitly."
}

$pythonPath = Resolve-PythonExecutable -Override $PythonExe
Write-Host "Using Python: $pythonPath"

Push-Location $projectDir
try {
    Write-Host "Environment smoke test..."
    & $pythonPath -X faulthandler -c "from runtime_env import configure_runtime_environment; configure_runtime_environment(); import io, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; print(np.eye(2) @ np.eye(2)); fig, ax = plt.subplots(); ax.plot([0, 1]); buffer = io.BytesIO(); fig.savefig(buffer, format='png'); print('PNG bytes:', len(buffer.getvalue()))"
    if ($LASTEXITCODE -ne 0) {
        throw "Environment smoke test failed with exit code $LASTEXITCODE"
    }

    $stages = @(
        "01_voxelize.py",
        "02_detect_seeds.py",
        "03_segment_3d.py",
        "04_export_traits.py"
    )

    foreach ($stage in $stages) {
        Write-Host "Running $stage ..."
        & $pythonPath -X faulthandler (Join-Path $projectDir $stage)
        if ($LASTEXITCODE -ne 0) {
            throw "$stage failed with exit code $LASTEXITCODE"
        }
    }

    Write-Host "Pipeline completed successfully."
    Write-Host "Results: $(Join-Path $projectDir 'results_ground_normalized')"
}
finally {
    Pop-Location
}
