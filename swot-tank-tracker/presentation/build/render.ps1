# Export every slide of the deck to PNG with PowerPoint, for visual checks.
# Usage: powershell -File render.ps1 [-Slides 3,5]
param([int[]]$Slides = @())
$root = Split-Path -Parent $PSScriptRoot
$deck = Join-Path $root "SWOT Tank Water Tracker - proposal talk.pptx"
$out = Join-Path $root "render"
New-Item -ItemType Directory -Force $out | Out-Null
$app = New-Object -ComObject PowerPoint.Application
try {
    $pres = $app.Presentations.Open($deck, $true, $false, $false)
    foreach ($slide in $pres.Slides) {
        if ($Slides.Count -eq 0 -or $Slides -contains $slide.SlideIndex) {
            $slide.Export((Join-Path $out ("slide-{0:D2}.png" -f $slide.SlideIndex)), "PNG", 1600, 900)
        }
    }
    $pres.Close()
} finally {
    $app.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($app) | Out-Null
}
Get-ChildItem $out -Filter "slide-*.png" | Select-Object -ExpandProperty Name
