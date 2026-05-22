$ErrorActionPreference = "Stop"

$usbipd = "usbipd"
$list = & $usbipd list
$line = $list | Select-String "3325:0049" | Select-Object -First 1
if (-not $line) {
    Write-Error "No MANUS dongle (3325:0049) found. Plug it in and retry."
}

$busid = ($line.ToString() -split "\s+")[0]
Write-Host "Sharing MANUS dongle on busid $busid"
& $usbipd bind --busid $busid
& $usbipd attach --wsl --busid $busid
Write-Host "Done. WSL should now see the MANUS dongle."
