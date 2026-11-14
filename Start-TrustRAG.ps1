$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Create the project virtual environment and install requirements-model.txt first. See README.md.'
}
& $pythonPath (Join-Path $projectRoot 'run.py') --model
