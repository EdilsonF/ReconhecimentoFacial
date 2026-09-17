Get-NetFirewallPortFilter | Where-Object { $_.Protocol -eq 'TCP' -and $_.LocalPort -ne 'Any' -and $_.LocalPort -match '^\d+$' } | Select-Object -ExpandProperty LocalPort | Sort-Object -Unique
