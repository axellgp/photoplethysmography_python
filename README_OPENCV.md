# Heart Rate Detector - Version OpenCV

## 🚀 Compatible Python 3.13.5

Cette version utilise **OpenCV** au lieu de MediaPipe pour la détection de visage, résolvant les problèmes de compatibilité avec Python 3.13.5.

## 📋 Différences avec la version MediaPipe

### ✅ Avantages
- **Compatible Python 3.13.5** - Pas de limitations de version
- **Plus simple à installer** - Moins de dépendances
- **Plus léger** - Moins de ressources utilisées
- **Plus stable** - Moins de risques de conflits

### ⚠️ Différences techniques
- **Détection de visage** : Utilise les cascades de Haar d'OpenCV
- **Précision** : Légèrement moins précise que MediaPipe pour la détection
- **Performance** : Peut être plus rapide sur certains systèmes

## 🛠️ Installation

### 1. Installation des dépendances
```bash
pip install -r requirements_opencv.txt
```

### 2. Lancement du programme
```bash
python heart_rate_detector_opencv.py
```

Ou utilisez le fichier batch :
```bash
LANCER_OPENCV.bat
```

## 🎯 Fonctionnalités

### Détection automatique
- Détection de visage avec OpenCV
- Extraction automatique de la zone du front
- Calcul du rythme cardiaque en temps réel

### Interface utilisateur
- **3 styles d'interface** : Minimal, Complet, Plein écran
- **Graphique BPM** en temps réel
- **Notifications** avec différents styles
- **Fenêtre redimensionnable**

### Modes de fonctionnement
- **Mode automatique** : Détection automatique du visage
- **Mode manuel** : Sélection manuelle de la zone d'intérêt

## 🎮 Contrôles

| Touche | Action |
|--------|--------|
| `A` | Mode automatique |
| `M` | Mode manuel |
| `V` | Heatmap vasculaire |
| `H` | Hotspots |
| `P` | Surcouche pulsante |
| `G` | Graphique BPM |
| `F` | Style UI |
| `N` | Style notifications |
| `R` | Reset |
| `ESC` | Quitter |

## 📊 Algorithme de détection

### 1. Détection de visage
```python
# Utilise les cascades de Haar d'OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
```

### 2. Extraction de la ROI
- Zone du front (tiers supérieur du visage)
- Filtrage et lissage du signal
- Calcul du BPM par analyse fréquentielle

### 3. Traitement du signal
- **Canal vert** : Plus sensible aux variations sanguines
- **Filtres** : Élimination des interférences (50Hz, 60Hz)
- **Lissage** : Moyenne mobile et interpolation

## 🔧 Configuration

### Paramètres modifiables
```python
# Dans la classe HeartRateDetectorOpenCV
self.bpm_thresholds = {
    'rest_min': 50,
    'rest_max': 90,
    'active_min': 90,
    'active_max': 150,
    'danger_min': 40,
    'danger_max': 200
}
```

### Optimisations
- **Cache** : Évite les calculs redondants
- **Buffers** : Lissage des valeurs BPM
- **Interpolation** : Graphiques plus fluides

## 🆚 Comparaison des versions

| Aspect | Version MediaPipe | Version OpenCV |
|--------|------------------|----------------|
| **Compatibilité Python** | 3.8-3.11 | 3.8-3.13.5 |
| **Taille installation** | ~500 MB | ~100 MB |
| **Précision détection** | Très haute | Haute |
| **Performance** | Rapide | Très rapide |
| **Stabilité** | Bonne | Excellente |
| **Facilité installation** | Complexe | Simple |

## 🎭 Styles d'interface

### Style Minimal
- BPM au centre uniquement
- Indicateur de mode discret
- Interface épurée

### Style Complet
- Toutes les informations affichées
- Statistiques détaillées
- Contrôles visibles

### Style Plein écran
- BPM géant au centre
- Graphique intégré
- Optimisé pour les grands écrans

## 🔔 Types de notifications

### Classique
- Rectangles simples
- Fond semi-transparent
- Texte blanc

### Moderne
- Bulles arrondies
- Icônes colorées
- Animations fluides

### Minimal
- Texte uniquement
- Couleurs d'accent
- Discret

## 📈 Analyse vasculaire

### Heatmap
- Visualisation des mouvements micro-vasculaires
- Superposition colorée
- Analyse temps réel

### Hotspots
- Points d'intérêt détectés
- Marqueurs jaunes
- Zones de pulsation

## 🎯 Conseils d'utilisation

### Pour de meilleurs résultats
1. **Éclairage** : Utilisez un éclairage uniforme
2. **Position** : Restez face à la caméra
3. **Stabilité** : Évitez les mouvements brusques
4. **Distance** : Placez-vous à 50-100 cm de la caméra

### Résolution des problèmes
- **Pas de détection** : Vérifiez l'éclairage et la position
- **BPM erratique** : Activez le mode manuel et sélectionnez une zone stable
- **Performance lente** : Réduisez la résolution de la caméra

## 📝 Notes de développement

### Structure du code
```
heart_rate_detector_opencv.py
├── OpenCVFaceDetector          # Détection de visage
├── OptimizedSignalProcessor    # Traitement du signal
├── HeartRateDetectorOpenCV     # Classe principale
└── main()                      # Point d'entrée
```

### Extensibilité
- Ajout facile de nouveaux styles
- Paramètres configurables
- Architecture modulaire

## 🆘 Support

### Problèmes courants
1. **Erreur d'importation** : Vérifiez l'installation des dépendances
2. **Caméra non détectée** : Vérifiez les permissions et pilotes
3. **BPM à zéro** : Assurez-vous que le visage est bien détecté

### Débogage
- Vérifiez les messages dans la console
- Utilisez le mode manuel si l'automatique ne fonctionne pas
- Testez avec différents styles d'interface

## 🎉 Conclusion

Cette version OpenCV offre une excellente alternative à MediaPipe, avec une meilleure compatibilité Python et des performances optimisées. Elle est particulièrement recommandée pour Python 3.13.5 et les environnements où MediaPipe pose des problèmes.

**Bonne utilisation ! 🚀**
