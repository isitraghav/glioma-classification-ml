# Convert a .pptx to PDF with PowerPoint. Usage: .pptx_to_pdf.ps1 -in <full path .pptx> -out <full path .pdf>
param($in, $out)
$pp = New-Object -ComObject PowerPoint.Application
$pres = $pp.Presentations.Open($in, $true, $false, $false)
$pres.SaveAs($out, 32)
$pres.Close()
$pp.Quit()
