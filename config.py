"""
Configuration pour le détecteur de rythme cardiaque rPPG
"""

# Configuration de la webcam
WEBCAM_CONFIG = {
    'fps': 30,
    'width': 640,
    'height': 480,
    'device_id': 0  # ID de la webcam (0 par défaut)
}

# Configuration du signal PPG
PPG_CONFIG = {
    'buffer_size': 300,  # Taille du buffer (~10s à 30 FPS)
    'min_buffer_fill': 0.8,  # Pourcentage minimum du buffer avant calcul
    'update_interval': 30,  # Nombre de frames entre les calculs de BPM
    'history_size': 10  # Nombre de BPM à moyenner
}

# Configuration du filtrage
FILTER_CONFIG = {
    'low_freq': 0.7,  # Fréquence basse (42 BPM)
    'high_freq': 4.0,  # Fréquence haute (240 BPM)
    'filter_order': 4,  # Ordre du filtre Butterworth
    'min_signal_length': 60  # Longueur minimale pour le filtrage
}

# Seuils BPM pour les couleurs
BPM_THRESHOLDS = {
    'normal_min': 60,
    'normal_max': 90,
    'critical_min': 50,
    'critical_max': 110
}

# Couleurs BGR
COLORS = {
    'normal': (0, 255, 0),      # Vert
    'warning': (0, 165, 255),   # Orange
    'critical': (0, 0, 255),    # Rouge
    'inactive': (128, 128, 128), # Gris
    'roi': (0, 255, 255),       # Jaune
    'text_bg': (0, 0, 0),       # Noir
    'text': (255, 255, 255)     # Blanc
}

# Configuration de l'affichage
DISPLAY_CONFIG = {
    'signal_overlay_width': 300,
    'signal_overlay_height': 100,
    'signal_display_length': 150,  # Nombre de points à afficher (~5s)
    'transparency': 0.7,
    'font_scale': 1.0,
    'font_thickness': 2
}

# Configuration MediaPipe
MEDIAPIPE_CONFIG = {
    'max_num_faces': 1,
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5,
    'refine_landmarks': True
}

# Points de repère pour la ROI du front
FOREHEAD_LANDMARKS = [
    10, 151, 9, 8, 337, 299, 333, 298, 301, 284, 251, 389, 356, 454, 
    323, 361, 340, 346, 347, 348, 349, 350, 451, 452, 453, 464, 435, 410
]

# Configuration de la stabilisation
STABILIZATION_CONFIG = {
    'center_history_size': 5,
    'stability_threshold': 20,  # Seuil de mouvement en pixels
    'adjustment_factor': 0.3,   # Facteur d'ajustement progressif
    'roi_height_factor': 0.6,   # Portion du front à utiliser
    'roi_width_factor': 0.25    # Largeur relative de la ROI
}
