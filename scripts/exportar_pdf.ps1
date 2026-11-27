param([string]$Documento = '', [string]$Destino = '')
$ErrorActionPreference = 'Stop'
if (-not $Documento) { $Documento = Join-Path $PSScriptRoot '../docs/informe_final.docx' }
$taskDocPath = (Resolve-Path -LiteralPath $Documento).Path
if (-not $Destino) { $Destino = [IO.Path]::ChangeExtension($taskDocPath, '.pdf') }
$taskPdfPath = [IO.Path]::GetFullPath($Destino)
$taskWord = $null; $taskDoc = $null
try {
    $taskWord = New-Object -ComObject Word.Application
    $taskWord.Visible = $false
    $taskWord.DisplayAlerts = 0
    $taskWord.AutomationSecurity = 3
    $taskDoc = $taskWord.Documents.Open($taskDocPath, $false, $true)
    $taskDoc.Repaginate()
    $taskPages = $taskDoc.ComputeStatistics(2)
    $taskDoc.ExportAsFixedFormat($taskPdfPath, 17)
    Write-Output "PDF generado: $taskPdfPath | Paginas: $taskPages"
} finally {
    if ($null -ne $taskDoc) { $taskDoc.Close(0) }
    if ($null -ne $taskWord) { $taskWord.Quit() }
}
