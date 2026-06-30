@echo off
title Person Locator Test Runner
echo --- Starte Person Locator Unit Tests ---
set PYTHONPATH=.;./personLocator
pytest tests
echo.
echo Testlauf abgeschlossen.
pause
