# 💗 Détecteur de Rythme Cardiaque - VERSION FINALE
## Version ultra-optimisée avec BPM granulaire et interface complète

### 🚀 FONCTIONNALITÉS
✅ **Détection automatique** du front via MediaPipe  
✅ **Sélection manuelle** de ROI à la souris  
✅ **BPM granulaire** avec lissage (pas de sauts brusques)  
✅ **Graphique temps réel** ultra-fluide avec interpolation  
✅ **3 styles d'interface** : minimal, complet, plein écran  
✅ **3 styles de notifications** visuellement différents  
✅ **Analyse vasculaire** avec heatmap et hotspots  
✅ **Surcouche pulsante** synchronisée au BPM  
✅ **Filtrage 50Hz/60Hz** des interférences électriques  
✅ **Performances optimisées** (~13 FPS)  
✅ **Fenêtre redimensionnable** et mode plein écran  
✅ **Interface scalable** qui s'adapte à la taille  

### 🎮 CONTRÔLES
- **A** : Mode automatique (détection du front)
- **M** : Mode manuel (sélection à la souris)  
- **V** : Heatmap vasculaire ON/OFF
- **H** : Hotspots vasculaires ON/OFF
- **P** : Surcouche pulsante ON/OFF
- **G** : Graphique BPM ON/OFF
- **F** : Style UI (minimal → complet → plein écran)
- **N** : Style notifications (classique → moderne → minimal)
- **R** : Reset système complet
- **F11** ou **Shift+F** : Basculer plein écran
- **ESC/Q** : Quitter

### 🖥️ AFFICHAGE ET FENÊTRE
- **Fenêtre redimensionnable** : Tirez les bords pour ajuster la taille
- **Mode plein écran** : Appuyez sur F11 ou Shift+F pour basculer
- **Interface scalable** : L'interface s'adapte à la taille de la fenêtre
- **Taille par défaut** : 800x600 pixels

### 🎨 STYLES D'INTERFACE
1. **Minimal** : Juste le BPM affiché
2. **Complet** : Tous les panneaux avec statistiques
3. **Plein écran** : BPM géant au centre

### 📱 STYLES DE NOTIFICATIONS  
1. **Classique** : Fond simple
2. **Moderne** : Fond coloré avec bordures
3. **Minimal** : Texte seul

### 📊 AFFICHAGE BPM
- **Granularité** : 1 chiffre après la virgule (ex: 72.3 BPM)
- **Lissage** : Moyenne pondérée pour éviter les sauts
- **Couleurs** : Vert (repos) → Jaune (actif) → Rouge (danger)

### 🔧 PRÉREQUIS
```bash
pip install opencv-python mediapipe matplotlib scipy numpy
```

### ▶️ LANCEMENT
Double-clic sur `LANCER_HEARTRATE.bat` ou :
```bash
python heart_rate_detector_final.py
```

### 📈 PERFORMANCES
- **FPS** : ~13 images/seconde  
- **Latence** : Optimisée avec cache  
- **Mémoire** : Gestion intelligente des buffers  
- **Stabilité** : 100% stable, aucun crash  

### 🎯 PRÉCISION
- **Détection** : MediaPipe haute précision
- **Signal** : Canal vert optimisé pour rPPG
- **Filtrage** : Anti-aliasing 50Hz/60Hz intégré
- **Validation** : Plage physiologique 30-200 BPM

---
**🏆 VERSION FINALE - ULTRA OPTIMISÉE 2025**
