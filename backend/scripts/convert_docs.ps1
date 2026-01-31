param(
    [string]$WordFolder = "$(Join-Path $PSScriptRoot '..' 'Word')"
)

if (!(Test-Path $WordFolder)) {
    Write-Error "Word folder not found: $WordFolder"
    exit 1
}

$libreOffice = "C:\Program Files\LibreOffice\program\soffice.exe"
if (!(Test-Path $libreOffice)) {
    Write-Error "LibreOffice not found at $libreOffice. Install LibreOffice or update the path in convert_docs.ps1."
    exit 1
}

$docs = Get-ChildItem -Path $WordFolder -Filter '*.doc' -Recurse
if ($docs.Count -eq 0) {
    Write-Host "No .doc files found under $WordFolder."
    exit 0
}

& $libreOffice --headless --convert-to pdf --outdir $WordFolder $docs.FullName
