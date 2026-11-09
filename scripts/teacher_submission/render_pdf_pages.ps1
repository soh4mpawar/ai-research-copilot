param(
    [string]$PdfPath = "C:\Users\soham\OneDrive\Desktop\rag\teacher_package\ONE_PAGE_SUMMARY.pdf",
    [string]$OutputDir = "C:\Users\soham\OneDrive\Desktop\rag\teacher_package\page_images_summary"
)

Add-Type -AssemblyName System.Runtime.WindowsRuntime
Add-Type -AssemblyName System.Drawing

$asTaskGeneric = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { 
    $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' 
} | Select-Object -First 1

function Await($WinRtTask, $ResultType) {
    $asTask = $asTaskGeneric.MakeGenericMethod($ResultType)
    $netTask = $asTask.Invoke($null, @($WinRtTask))
    $netTask.Wait(-1) | Out-Null
    return $netTask.Result
}

function AwaitAction($WinRtAction) {
    $asTaskAction = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { 
        $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncAction' 
    } | Select-Object -First 1
    $netTask = $asTaskAction.Invoke($null, @($WinRtAction))
    $netTask.Wait(-1) | Out-Null
}

[Windows.Data.Pdf.PdfDocument, Windows.Data.Pdf, ContentType = WindowsRuntime] | Out-Null
[Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime] | Out-Null
[Windows.Storage.Streams.InMemoryRandomAccessStream, Windows.Storage.Streams, ContentType = WindowsRuntime] | Out-Null

if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

$resolvedPdf = (Resolve-Path $PdfPath).Path
$storageFile = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($resolvedPdf)) ([Windows.Storage.StorageFile])
$pdfDoc = Await ([Windows.Data.Pdf.PdfDocument]::LoadFromFileAsync($storageFile)) ([Windows.Data.Pdf.PdfDocument])

Write-Host "Rendering $($pdfDoc.PageCount) pages from $PdfPath to $OutputDir..."

for ($i = 0; $i -lt $pdfDoc.PageCount; $i++) {
    $page = $pdfDoc.GetPage($i)
    $stream = New-Object Windows.Storage.Streams.InMemoryRandomAccessStream
    AwaitAction ($page.RenderToStreamAsync($stream))
    
    # Read stream into .NET MemoryStream
    $netStream = New-Object System.IO.MemoryStream
    $bytes = New-Object byte[] $stream.Size
    $reader = New-Object Windows.Storage.Streams.DataReader($stream.GetInputStreamAt(0))
    Await ($reader.LoadAsync($stream.Size)) ([System.UInt32]) | Out-Null
    $reader.ReadBytes($bytes)
    $reader.Dispose()
    
    $netStream.Write($bytes, 0, $bytes.Length)
    $img = [System.Drawing.Image]::FromStream($netStream)
    $pageOut = Join-Path $OutputDir ("page_{0:D2}.png" -f ($i + 1))
    $img.Save($pageOut, [System.Drawing.Imaging.ImageFormat]::Png)
    $img.Dispose()
    $netStream.Dispose()
    $stream.Dispose()
    Write-Host "Saved page $($i + 1) to $pageOut"
}
Write-Host "Rendering complete!"
