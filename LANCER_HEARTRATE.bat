@echo off
title 💗 Heart Rate Monitor - VERSION FINALE
color 0A

echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║         💗 DÉTECTEUR DE RYTHME CARDIAQUE 💗            ║
echo  ║              VERSION FINALE OPTIMISÉE                   ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.
echo  🚀 Lancement de l'application...
echo  📊 BPM granulaire et graphique fluide
echo  🎨 Interface complète avec 3 styles
echo  🔧 Performances optimisées (~13 FPS)
echo.
echo  🎮 CONTRÔLES RAPIDES:
echo    A = Mode auto   │  M = Mode manuel   │  F = Style UI
echo    V = Heatmap     │  H = Hotspots      │  G = Graphique  
echo    P = Pulsation   │  N = Notifications │  R = Reset
echo    Q = Quitter     │  ESC = Quitter
echo.

cd /d "%~dp0"

if not exist "heart_rate_detector_final.py" (
    echo ❌ ERREUR: Fichier heart_rate_detector_final.py introuvable!
    echo    Vérifiez que tous les fichiers sont présents.
    pause
    exit /b 1
)

echo ⚡ Démarrage...
python heart_rate_detector_final.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ ERREUR lors de l'exécution!
    echo 🔧 Vérifiez que Python et les dépendances sont installées:
    echo    pip install opencv-python mediapipe matplotlib scipy numpy
    echo.
    pause
) else (
    echo.
    echo ✅ Application fermée normalement.
    echo 📊 Merci d'avoir utilisé Heart Rate Monitor!
    timeout /t 3 >nul
)
