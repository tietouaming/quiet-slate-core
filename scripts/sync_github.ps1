param(
    [Parameter(Mandatory = $false)]
    [string]$Message = ""
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path ".git")) {
    throw "Current directory is not a git repository."
}

if ([string]::IsNullOrWhiteSpace($Message)) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $Message = "chore: sync $ts"
}

git add -A

$status = git status --porcelain
if ([string]::IsNullOrWhiteSpace($status)) {
    Write-Output "No changes to commit."
    exit 0
}

git commit -m $Message
git push

Write-Output "Sync complete."
