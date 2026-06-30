# PowerShell Script zum automatischen Ausführen der Tests
Write-Host "--- Starte Person Locator Unit Tests ---" -ForegroundColor Cyan

# Setze Umgebung (falls pytest.ini nicht reicht oder manuell aufgerufen wird)
$env:PYTHONPATH = ".;./personLocator"

# Führe pytest aus
pytest tests

Write-Host "`nTestlauf abgeschlossen. Drücke eine Taste zum Beenden..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
