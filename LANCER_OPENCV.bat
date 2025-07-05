@echo off
echo ========================================================
echo   HEART RATE DETECTOR - VERSION OPENCV
echo   Compatible Python 3.13.5 (sans MediaPipe)
echo ========================================================
echo.

echo Verification de Python...
python --version
echo.

echo Lancement du detecteur de rythme cardiaque...
echo.
echo CONTROLES :
echo   'A' : Mode automatique
echo   'M' : Mode manuel  
echo   'V' : Heatmap vasculaire
echo   'G' : Graphique BPM
echo   'F' : Style UI
echo   'R' : Reset
echo   'ESC' : Quitter
echo.

python heart_rate_detector_opencv.py

echo.
echo Programme termine. Appuyez sur une touche pour fermer...
pause >nul
