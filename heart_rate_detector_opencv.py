"""
Détecteur de rythme cardiaque OPTIMISÉ - VERSION OPENCV (Sans MediaPipe)
===========================================================

Version alternative utilisant OpenCV avec cascades de Haar pour la détection de visage.
Compatible avec Python 3.13.5 - Pas de dépendance MediaPipe.

✅ Détection de visage avec OpenCV (Haar Cascades)
✅ BPM ultra-granulaire et lissé
✅ Graphique BPM fluide
✅ Styles UI différents et effectifs
✅ Notifications avec styles visuels distincts
✅ Performances optimisées
✅ Calcul BPM précis avec interpolation
✅ Interface réactive et stable

Contrôles :
- 'A' : Mode automatique
- 'M' : Mode manuel
- 'V' : Heatmap vasculaire
- 'H' : Hotspots
- 'S' : ROI suggérée
- 'P' : Surcouche pulsante
- 'G' : Graphique BPM
- 'F' : Style UI (minimal/complet/plein écran)
- 'N' : Style notifications (classique/moderne/minimal)
- 'R' : Reset
- 'F11' ou 'Shift+F' : Basculer plein écran
- 'ESC'/'Q' : Quitter

Fonctionnalités :
- Fenêtre redimensionnable (tirez les bords)
- Mode plein écran avec F11 ou Shift+F
- Interface scalable qui s'adapte à la taille de la fenêtre
- Compatible Python 3.13.5
"""

import cv2
import numpy as np
from collections import deque
import time
import sys
import os

# Essayer d'importer scipy, sinon utiliser des alternatives
try:
    from scipy.signal import butter, filtfilt
    from scipy.interpolate import interp1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("⚠️  Scipy non disponible - utilisation des alternatives intégrées")

# Ajouter le dossier courant au path pour importer nos modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import SignalProcessor
from utils_manual import ManualROISelector, HeartRateVisualizer
from vascular_analyzer_simple import VascularMicroMovementAnalyzer
import config


class OpenCVFaceDetector:
    """
    Détecteur de visage utilisant OpenCV au lieu de MediaPipe
    """
    
    def __init__(self):
        # Charger les cascades de Haar
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Paramètres de détection
        self.scale_factor = 1.1
        self.min_neighbors = 5
        self.min_size = (30, 30)
        
        # Cache pour stabiliser la détection
        self.face_cache = deque(maxlen=10)
        self.last_stable_face = None
        
    def detect_face(self, frame):
        """Détecte le visage dans une image"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Détection des visages
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size
        )
        
        if len(faces) > 0:
            # Prendre le plus grand visage
            face = max(faces, key=lambda f: f[2] * f[3])
            self.face_cache.append(face)
            self.last_stable_face = face
            return face
        
        # Si pas de visage détecté, utiliser le cache
        if self.last_stable_face is not None:
            return self.last_stable_face
        
        return None
    
    def get_forehead_roi(self, frame, face_box):
        """Extrait la ROI du front à partir de la boîte du visage"""
        if face_box is None:
            return None
        
        x, y, w, h = face_box
        
        # Définir la zone du front (tiers supérieur du visage)
        forehead_x = x + int(w * 0.2)
        forehead_y = y + int(h * 0.1)
        forehead_w = int(w * 0.6)
        forehead_h = int(h * 0.3)
        
        # Vérifier que la ROI est dans les limites de l'image
        frame_h, frame_w = frame.shape[:2]
        if (forehead_x + forehead_w > frame_w or 
            forehead_y + forehead_h > frame_h or 
            forehead_x < 0 or forehead_y < 0):
            return None
        
        return (forehead_x, forehead_y, forehead_w, forehead_h)


class OptimizedSignalProcessor:
    """
    Processeur de signal optimisé avec BPM ultra-granulaire
    """
    
    def __init__(self, fps=30, window_size=150):
        self.fps = fps
        self.window_size = window_size
        self.signal_buffer = deque(maxlen=window_size)
        self.signal_processor = SignalProcessor(fps=fps)
        
        # Buffers pour le lissage BPM
        self.bpm_buffer = deque(maxlen=10)
        self.raw_bpm_buffer = deque(maxlen=30)
        
        # Cache pour optimiser les calculs
        self.last_bpm_time = 0
        self.cached_bpm = 0
        self.bpm_cache_duration = 0.1  # 100ms de cache
        
        # Filtres pour éliminer les interférences électriques
        self.nyquist = fps / 2
        self.notch_filters = self._create_notch_filters()
        
    def _create_notch_filters(self):
        """Crée les filtres coupe-bande pour 50Hz et 60Hz"""
        filters = []
        
        # Filtre pour 50Hz (Europe)
        if 50 < self.nyquist:
            low_50 = (50 - 2) / self.nyquist
            high_50 = (50 + 2) / self.nyquist
            if 0 < low_50 < 1 and 0 < high_50 < 1:
                filters.append(('50Hz', butter(2, [low_50, high_50], btype='bandstop')))
        
        # Filtre pour 60Hz (Amérique)
        if 60 < self.nyquist:
            low_60 = (60 - 2) / self.nyquist
            high_60 = (60 + 2) / self.nyquist
            if 0 < low_60 < 1 and 0 < high_60 < 1:
                filters.append(('60Hz', butter(2, [low_60, high_60], btype='bandstop')))
        
        return filters
    
    def add_signal(self, signal_value):
        """Ajoute une valeur de signal"""
        self.signal_buffer.append(signal_value)
        
    def get_smooth_bpm(self):
        """Calcule un BPM lissé et granulaire"""
        current_time = time.time()
        
        # Utiliser le cache si récent
        if current_time - self.last_bpm_time < self.bpm_cache_duration:
            return self.cached_bpm
        
        if len(self.signal_buffer) < self.window_size // 2:
            return 0
        
        # Convertir en array numpy
        signal_array = np.array(self.signal_buffer)
        
        # Appliquer les filtres anti-interférences
        filtered_signal = signal_array.copy()
        for name, (b, a) in self.notch_filters:
            try:
                filtered_signal = filtfilt(b, a, filtered_signal)
            except:
                pass  # Ignorer les erreurs de filtrage
        
        # Calcul BPM avec le processeur standard
        bpm = self.signal_processor.calculate_bpm_from_fft(filtered_signal.tolist())
        
        if bpm > 0:
            self.raw_bpm_buffer.append(bpm)
            
            # Lissage avec médiane glissante
            if len(self.raw_bpm_buffer) >= 5:
                recent_bpms = list(self.raw_bpm_buffer)[-5:]
                median_bpm = np.median(recent_bpms)
                
                # Interpolation pour plus de granularité
                self.bpm_buffer.append(median_bpm)
                
                if len(self.bpm_buffer) >= 3:
                    # Moyenne pondérée des dernières valeurs
                    weights = np.array([0.5, 0.3, 0.2])
                    if len(self.bpm_buffer) >= 3:
                        recent_values = np.array(list(self.bpm_buffer)[-3:])
                        smooth_bpm = np.average(recent_values, weights=weights)
                    else:
                        smooth_bpm = median_bpm
                    
                    self.cached_bpm = smooth_bpm
                    self.last_bpm_time = current_time
                    return smooth_bpm
        
        return self.cached_bpm


class HeartRateDetectorOpenCV:
    """
    Détecteur de rythme cardiaque optimisé utilisant OpenCV
    """
    
    def __init__(self):
        # Initialisation des composants
        self.signal_processor = OptimizedSignalProcessor(fps=30)
        self.manual_roi_selector = ManualROISelector()
        self.heart_rate_visualizer = HeartRateVisualizer()
        self.vascular_analyzer = VascularMicroMovementAnalyzer(
            history_size=5, sensitivity=0.1  # Réduit pour les performances
        )
        
        # OpenCV pour la détection du visage
        self.face_detector = OpenCVFaceDetector()
        
        # Variables d'état
        self.mode = "auto"  # "auto" ou "manual"
        self.current_roi = None
        self.is_selecting_roi = False
        self.is_fullscreen = False  # Variable pour le mode plein écran
        
        # Métriques et historique
        self.current_bpm = 0
        self.bpm_history = deque(maxlen=100)  # Plus de points pour le graphique
        self.detailed_bpm_history = deque(maxlen=300)  # Historique détaillé
        self.avg_bpm_history = deque(maxlen=50)
        self.signal_quality_history = deque(maxlen=30)
        
        # Compteurs
        self.frame_count = 0
        self.face_detection_count = 0
        self.start_time = time.time()
        
        # Options d'affichage
        self.show_vascular_heatmap = False
        self.show_vascular_hotspots = False
        self.show_analysis_info = False
        self.show_pulse_overlay = True
        self.show_bpm_graph = True
        self.show_suggested_roi = False
        self.show_detailed_stats = False
        self.show_performance_logs = False
        self.show_signal_quality = True
        
        # Styles UI (maintenant vraiment différents)
        self.ui_styles = ["minimal", "complete", "fullscreen"]
        self.ui_style = "complete"
        
        # Styles de notification (maintenant visuellement différents)
        self.notification_styles = ["classic", "modern", "minimal"]
        self.notification_style = "modern"
        
        # Variables pour les notifications
        self.notification_queue = deque(maxlen=5)
        self.notification_timer = 0
        
        # Sensibilité vasculaire
        self.vascular_sensitivity_levels = [0.05, 0.1, 0.2, 0.3]
        self.current_sensitivity_index = 1
        
        # Thresholds configurables
        self.bpm_threshold_mode = 0  # 0: normal, 1: sport, 2: repos, 3: personnalisé
        self.threshold_modes = ["Normal", "Sport", "Repos", "Personnalisé"]
        
        # Logs de performance
        self.performance_logs = deque(maxlen=100)
        self.fps_history = deque(maxlen=30)
        
        # Qualité du signal
        self.signal_quality = 0
        self.last_fps_time = time.time()
        self.fps_counter = 0
        
        # Couleurs
        self.colors = {
            'good': (0, 255, 0),
            'warning': (0, 255, 255),
            'danger': (0, 0, 255),
            'inactive': (128, 128, 128),
            'text': (255, 255, 255),
            'accent': (255, 100, 100),
            'bg': (0, 0, 0)
        }
        
        # Seuils BPM
        self.bpm_thresholds = {
            'rest_min': 50,
            'rest_max': 90,
            'active_min': 90,
            'active_max': 150,
            'danger_min': 40,
            'danger_max': 200
        }
        
        # Cache pour les calculs lourds
        self.roi_cache = {}
        self.last_roi_time = 0
        
        # Gestion plein écran
        self.window_name = 'Heart Rate Detector - OpenCV'
        
    def update_bpm_thresholds(self):
        """Met à jour les seuils BPM selon le mode"""
        if self.bmp_threshold_mode == 0:  # Normal
            self.bpm_thresholds.update({
                'rest_min': 50, 'rest_max': 90,
                'active_min': 90, 'active_max': 150,
                'danger_min': 40, 'danger_max': 200
            })
        elif self.bpm_threshold_mode == 1:  # Sport
            self.bpm_thresholds.update({
                'rest_min': 60, 'rest_max': 100,
                'active_min': 100, 'active_max': 180,
                'danger_min': 50, 'danger_max': 220
            })
        elif self.bpm_threshold_mode == 2:  # Repos
            self.bpm_thresholds.update({
                'rest_min': 45, 'rest_max': 80,
                'active_min': 80, 'active_max': 120,
                'danger_min': 35, 'danger_max': 150
            })
        # Mode 3 (personnalisé) : pas de changement automatique
    
    def calculate_signal_quality(self):
        """Calcule la qualité du signal"""
        if len(self.signal_processor.signal_buffer) < 30:
            return 0
        
        signal = np.array(list(self.signal_processor.signal_buffer)[-30:])
        
        # Variabilité du signal
        signal_std = np.std(signal)
        signal_mean = np.mean(signal)
        
        # Ratio signal/bruit
        if signal_mean > 0:
            snr = signal_std / signal_mean
        else:
            snr = 0
        
        # Score de qualité (0-100)
        quality = min(100, max(0, snr * 100))
        
        return quality
    
    def add_performance_log(self, action, duration=None):
        """Ajoute un log de performance"""
        log_entry = {
            'time': time.time(),
            'action': action,
            'duration': duration,
            'bpm': self.current_bpm,
            'quality': self.signal_quality
        }
        self.performance_logs.append(log_entry)
    
    def save_session_data(self):
        """Sauvegarde les données de la session"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"heart_rate_session_{timestamp}.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("=== SESSION HEART RATE DETECTOR ===\n")
                f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Durée: {time.time() - self.start_time:.1f}s\n")
                f.write(f"Frames: {self.frame_count}\n")
                f.write(f"Détections: {self.face_detection_count}\n\n")
                
                f.write("=== DONNÉES BPM ===\n")
                for i, bpm in enumerate(self.bpm_history):
                    f.write(f"{i},{bpm:.2f}\n")
                
                f.write("\n=== LOGS PERFORMANCE ===\n")
                for log in self.performance_logs:
                    f.write(f"{log['time']},{log['action']},{log['bpm']},{log['quality']}\n")
            
            self.add_notification(f"Données sauvées: {filename}", "success")
            return True
        except Exception as e:
            self.add_notification(f"Erreur sauvegarde: {str(e)}", "error")
            return False
        
    def add_notification(self, message, type="info"):
        """Ajoute une notification à la queue"""
        self.notification_queue.append({
            'message': message,
            'type': type,
            'time': time.time()
        })
        
    def detect_forehead_roi(self, frame):
        """Détecte automatiquement la ROI du front (avec cache)"""
        current_time = time.time()
        
        # Utiliser le cache si récent (optimisation)
        if current_time - self.last_roi_time < 0.5:  # 500ms de cache
            return self.roi_cache.get('last_roi', None)
        
        # Détecter le visage
        face_box = self.face_detector.detect_face(frame)
        
        if face_box is not None:
            # Obtenir la ROI du front
            forehead_roi = self.face_detector.get_forehead_roi(frame, face_box)
            
            if forehead_roi is not None:
                self.roi_cache['last_roi'] = forehead_roi
                self.last_roi_time = current_time
                return forehead_roi
        
        return self.roi_cache.get('last_roi', None)
    
    def get_bpm_color(self, bpm):
        """Détermine la couleur selon le BPM"""
        if bpm <= 0:
            return self.colors['inactive']
        elif bpm < self.bpm_thresholds['danger_min'] or bpm > self.bpm_thresholds['danger_max']:
            return self.colors['danger']
        elif bpm < self.bpm_thresholds['rest_min'] or bpm > self.bpm_thresholds['active_max']:
            return self.colors['warning']
        else:
            return self.colors['good']
    
    def create_bpm_graph(self, width=400, height=200):
        """Crée un graphique BPM avec OpenCV (plus simple et compatible)"""
        if len(self.detailed_bpm_history) < 10:
            return None
        
        # Créer une image noire
        graph = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Préparer les données
        bpm_data = list(self.detailed_bpm_history)
        
        if len(bpm_data) < 2:
            return graph
        
        # Normaliser les données
        min_bpm = max(40, min(bpm_data) - 10)
        max_bpm = min(200, max(bpm_data) + 10)
        
        if max_bpm <= min_bpm:
            max_bpm = min_bpm + 50
        
        # Convertir en coordonnées pixel
        points = []
        for i, bpm in enumerate(bpm_data):
            x = int((i / (len(bpm_data) - 1)) * (width - 40)) + 20
            y = int(height - 20 - ((bpm - min_bpm) / (max_bpm - min_bpm)) * (height - 40))
            points.append((x, y))
        
        # Dessiner la ligne
        for i in range(len(points) - 1):
            cv2.line(graph, points[i], points[i + 1], (0, 255, 0), 2)
        
        # Dessiner les points
        for point in points:
            cv2.circle(graph, point, 2, (0, 255, 0), -1)
        
        # Ajouter les axes et labels
        cv2.line(graph, (20, 20), (20, height - 20), (100, 100, 100), 1)  # Axe Y
        cv2.line(graph, (20, height - 20), (width - 20, height - 20), (100, 100, 100), 1)  # Axe X
        
        # Labels
        cv2.putText(graph, 'BPM', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(graph, f'{max_bpm:.0f}', (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.putText(graph, f'{min_bpm:.0f}', (25, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Ligne moyenne
        if len(bpm_data) > 0:
            avg_bpm = np.mean(bpm_data)
            y_avg = int(height - 20 - ((avg_bpm - min_bpm) / (max_bpm - min_bpm)) * (height - 40))
            cv2.line(graph, (20, y_avg), (width - 20, y_avg), (255, 255, 255), 1)
            cv2.putText(graph, f'Moy: {avg_bpm:.1f}', (width - 80, y_avg - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return graph
    
    def draw_notifications(self, frame):
        """Dessine les notifications avec différents styles"""
        if not self.notification_queue:
            return
        
        h, w = frame.shape[:2]
        current_time = time.time()
        
        # Nettoyer les notifications expirées
        while self.notification_queue and current_time - self.notification_queue[0]['time'] > 3:
            self.notification_queue.popleft()
        
        if not self.notification_queue:
            return
        
        # Style des notifications
        if self.notification_style == "classic":
            # Style classique - rectangles simples
            for i, notif in enumerate(self.notification_queue):
                age = current_time - notif['time']
                alpha = max(0, 1 - age / 3)  # Fade out
                
                y_pos = 100 + i * 40
                
                # Fond
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, y_pos - 25), (400, y_pos + 10), (0, 0, 0), -1)
                cv2.addWeighted(frame, 1 - alpha * 0.7, overlay, alpha * 0.7, 0, frame)
                
                # Texte
                color = self.colors['text'] if alpha > 0.5 else tuple(int(c * alpha) for c in self.colors['text'])
                cv2.putText(frame, notif['message'], (20, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        elif self.notification_style == "modern":
            # Style moderne - bulles arrondies
            for i, notif in enumerate(self.notification_queue):
                age = current_time - notif['time']
                alpha = max(0, 1 - age / 3)
                
                y_pos = 100 + i * 45
                
                # Fond arrondi (simulation)
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, y_pos - 30), (450, y_pos + 15), (40, 40, 40), -1)
                cv2.rectangle(overlay, (15, y_pos - 25), (445, y_pos + 10), (60, 60, 60), -1)
                cv2.addWeighted(frame, 1 - alpha * 0.8, overlay, alpha * 0.8, 0, frame)
                
                # Icône selon le type
                icon_color = self.colors['good'] if notif['type'] == 'success' else self.colors['warning']
                cv2.circle(frame, (35, y_pos - 8), 8, icon_color, -1)
                
                # Texte
                color = self.colors['text'] if alpha > 0.5 else tuple(int(c * alpha) for c in self.colors['text'])
                cv2.putText(frame, notif['message'], (55, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        elif self.notification_style == "minimal":
            # Style minimal - juste le texte
            for i, notif in enumerate(self.notification_queue):
                age = current_time - notif['time']
                alpha = max(0, 1 - age / 3)
                
                y_pos = 100 + i * 25
                
                # Texte seulement
                color = self.colors['accent'] if alpha > 0.7 else tuple(int(c * alpha) for c in self.colors['accent'])
                cv2.putText(frame, notif['message'], (20, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def draw_ui(self, frame):
        """Dessine l'interface utilisateur selon le style choisi"""
        h, w = frame.shape[:2]
        
        if self.ui_style == "minimal":
            # Style minimal - juste le BPM
            bpm_color = self.get_bpm_color(self.current_bpm)
            bpm_text = f"{self.current_bpm:.1f}" if self.current_bpm > 0 else "--"
            
            # BPM centré
            font_scale = min(w / 600, h / 400)
            text_size = cv2.getTextSize(bpm_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 3, 4)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h // 2
            
            cv2.putText(frame, bpm_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale * 3, bpm_color, 4)
            
            # Petit indicateur du mode
            mode_text = f"[{self.mode.upper()}]"
            cv2.putText(frame, mode_text, (w - 100, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        elif self.ui_style == "complete":
            # Style complet - toutes les informations
            
            # BPM principal
            bpm_color = self.get_bpm_color(self.current_bpm)
            bpm_text = f"BPM: {self.current_bpm:.1f}" if self.current_bpm > 0 else "BPM: Calcul..."
            cv2.putText(frame, bpm_text, (15, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, bpm_color, 3)
            
            # Informations détaillées
            info_y = 100
            info_texts = [
                f"Mode: {self.mode.upper()}",
                f"Frames: {self.frame_count}",
                f"Détections: {self.face_detection_count}",
                f"Qualité: {len(self.signal_quality_history)}/30"
            ]
            
            for text in info_texts:
                cv2.putText(frame, text, (15, info_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 1)
                info_y += 25
            
            # Moyenne BPM
            if len(self.avg_bpm_history) > 0:
                avg_bpm = np.mean(self.avg_bpm_history)
                cv2.putText(frame, f"Moyenne: {avg_bpm:.1f}", (15, info_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['accent'], 1)
            
            # Contrôles
            controls_y = h - 120
            controls = [
                "'A': Auto  'M': Manuel  'V': Vascular",
                "'G': Graphique  'F': Style UI",
                "'R': Reset  'ESC': Quitter"
            ]
            
            for text in controls:
                cv2.putText(frame, text, (15, controls_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
                controls_y += 20
        
        elif self.ui_style == "fullscreen":
            # Style plein écran - BPM géant au centre
            bpm_color = self.get_bpm_color(self.current_bpm)
            bpm_text = f"{self.current_bpm:.2f}" if self.current_bpm > 0 else "--"
            
            # Taille de police adaptative
            font_scale = min(w / 400, h / 300)
            text_size = cv2.getTextSize(bpm_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 2, 4)[0]
            text_x = (w - text_size[0]) // 2
            text_y = (h + text_size[1]) // 2
            
            cv2.putText(frame, bpm_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale * 2, bpm_color, 4)
            
            # Indicateur de statut minimal
            status_text = f"[{self.mode}] {self.face_detection_count} détections"
            cv2.putText(frame, status_text, (20, h - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 1)
        
        # Graphique BPM fluide
        if self.show_bpm_graph and len(self.detailed_bpm_history) > 10:
            graph_img = self.create_bpm_graph(width=min(400, w//2), height=min(200, h//3))
            if graph_img is not None:
                graph_h, graph_w = graph_img.shape[:2]
                
                # Position du graphique
                if self.ui_style == "fullscreen":
                    x_pos = w - graph_w - 20
                    y_pos = 20
                else:
                    x_pos = w - graph_w - 15
                    y_pos = 15
                
                # Vérifier les limites
                if x_pos > 0 and y_pos > 0 and x_pos + graph_w < w and y_pos + graph_h < h:
                    frame[y_pos:y_pos+graph_h, x_pos:x_pos+graph_w] = graph_img
        
        # Dessiner les notifications
        self.draw_notifications(frame)
    
    def process_frame(self, frame):
        """Traite une frame et calcule le BPM"""
        self.frame_count += 1
        
        # Sélection de la ROI
        if self.mode == "auto":
            roi = self.detect_forehead_roi(frame)
            if roi is not None:
                self.current_roi = roi
                self.face_detection_count += 1
        elif self.mode == "manual":
            if self.is_selecting_roi:
                roi = self.manual_roi_selector.get_selection(frame)
                if roi is not None:
                    self.current_roi = roi
                    self.is_selecting_roi = False
        
        # Traitement du signal si on a une ROI
        if self.current_roi is not None:
            x, y, w, h = self.current_roi
            
            # Extraire la ROI
            roi_frame = frame[y:y+h, x:x+w]
            
            # Calculer la valeur moyenne du signal
            if roi_frame.size > 0:
                # Utiliser le canal vert (plus sensible aux variations sanguines)
                green_channel = roi_frame[:, :, 1]
                signal_value = np.mean(green_channel)
                
                # Ajouter au processeur de signal
                self.signal_processor.add_signal(signal_value)
                
                # Calculer le BPM
                new_bpm = self.signal_processor.get_smooth_bpm()
                
                if new_bpm > 0:
                    self.current_bpm = new_bpm
                    self.bpm_history.append(new_bpm)
                    self.detailed_bpm_history.append(new_bpm)
                    
                    # Moyenne mobile pour l'historique
                    if len(self.bpm_history) >= 10:
                        avg_bpm = np.mean(list(self.bpm_history)[-10:])
                        self.avg_bpm_history.append(avg_bpm)
                
                # Dessiner la ROI
                color = self.get_bpm_color(self.current_bpm)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Overlay pulsant
                if self.show_pulse_overlay and self.current_bpm > 0:
                    pulse_intensity = abs(np.sin(time.time() * self.current_bpm / 60 * 2 * np.pi))
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x, y), (x+w, y+h), color, -1)
                    cv2.addWeighted(frame, 1 - pulse_intensity * 0.3, overlay, pulse_intensity * 0.3, 0, frame)
        
        # Analyse vasculaire
        if self.show_vascular_heatmap or self.show_vascular_hotspots:
            self.vascular_analyzer.analyze_frame(frame)
            
            if self.show_vascular_heatmap:
                heatmap = self.vascular_analyzer.get_heatmap()
                if heatmap is not None:
                    overlay = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)
            
            if self.show_vascular_hotspots:
                hotspots = self.vascular_analyzer.get_hotspots()
                for (x, y) in hotspots:
                    cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
        
        # Dessiner l'interface utilisateur
        self.draw_ui(frame)
        
        return frame
    
    def handle_key(self, key):
        """Gère les touches du clavier"""
        if key == ord('a') or key == ord('A'):
            self.mode = "auto"
            self.add_notification("Mode automatique activé", "success")
        
        elif key == ord('m') or key == ord('M'):
            self.mode = "manual"
            self.is_selecting_roi = True
            self.add_notification("Mode manuel - Sélectionnez une zone", "info")
        
        elif key == ord('v') or key == ord('V'):
            self.show_vascular_heatmap = not self.show_vascular_heatmap
            status = "activée" if self.show_vascular_heatmap else "désactivée"
            self.add_notification(f"Heatmap vasculaire {status}", "info")
        
        elif key == ord('h') or key == ord('H'):
            self.show_vascular_hotspots = not self.show_vascular_hotspots
            status = "activés" if self.show_vascular_hotspots else "désactivés"
            self.add_notification(f"Hotspots {status}", "info")
        
        elif key == ord('p') or key == ord('P'):
            self.show_pulse_overlay = not self.show_pulse_overlay
            status = "activée" if self.show_pulse_overlay else "désactivée"
            self.add_notification(f"Surcouche pulsante {status}", "info")
        
        elif key == ord('g') or key == ord('G'):
            self.show_bpm_graph = not self.show_bpm_graph
            status = "activé" if self.show_bpm_graph else "désactivé"
            self.add_notification(f"Graphique BPM {status}", "info")
        
        elif key == ord('f') or key == ord('F'):
            current_index = self.ui_styles.index(self.ui_style)
            self.ui_style = self.ui_styles[(current_index + 1) % len(self.ui_styles)]
            self.add_notification(f"Style UI: {self.ui_style}", "info")
        
        elif key == ord('n') or key == ord('N'):
            current_index = self.notification_styles.index(self.notification_style)
            self.notification_style = self.notification_styles[(current_index + 1) % len(self.notification_styles)]
            self.add_notification(f"Style notifications: {self.notification_style}", "info")
        
        elif key == ord('r') or key == ord('R'):
            self.reset()
            self.add_notification("Système réinitialisé", "success")
        
        elif key == 27 or key == ord('q') or key == ord('Q'):  # ESC ou Q
            return False
        
        return True
    
    def reset(self):
        """Remet à zéro tous les compteurs et historiques"""
        self.current_bpm = 0
        self.bpm_history.clear()
        self.detailed_bpm_history.clear()
        self.avg_bpm_history.clear()
        self.signal_quality_history.clear()
        self.frame_count = 0
        self.face_detection_count = 0
        self.current_roi = None
        self.signal_processor = OptimizedSignalProcessor(fps=30)
        self.start_time = time.time()


def main():
    """Fonction principale"""
    print("🚀 Détecteur de Rythme Cardiaque - Version OpenCV")
    print("=" * 55)
    print()
    print("✅ Compatible Python 3.13.5")
    print("✅ Utilise OpenCV (pas de MediaPipe)")
    print("✅ Détection de visage avec Haar Cascades")
    print("✅ Interface ultra-fluide et réactive")
    print()
    print("🎯 CONTRÔLES PRINCIPAUX :")
    print("🔄 'A' : Mode automatique")
    print("✋ 'M' : Mode manuel")
    print("🔬 'V' : Heatmap vasculaire")
    print("🎯 'H' : Hotspots")
    print("💓 'P' : Surcouche pulsante")
    print("📊 'G' : Graphique BPM")
    print("🎨 'F' : Style UI (minimal/complet/plein écran)")
    print("🔔 'N' : Style notifications")
    print("🔄 'R' : Reset")
    print("🖥️ 'F11' ou 'Shift+F' : Basculer plein écran")
    print("🚪 'ESC' ou 'Q' : Quitter")
    print()
    print("💡 ASTUCE : La fenêtre est redimensionnable - tirez les bords pour l'agrandir !")
    print("🖼️ CONSEIL : Utilisez F11 ou Shift+F pour passer en plein écran")
    print()
    
    # Initialiser le détecteur
    detector = HeartRateDetectorOpenCV()
    
    # Initialiser la caméra
    cap = cv2.VideoCapture(0)
    
    # Configuration de la caméra
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Créer la fenêtre
    cv2.namedWindow('Heart Rate Detector - OpenCV', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Heart Rate Detector - OpenCV', 800, 600)
    
    print("🎥 Caméra initialisée - Appuyez sur 'ESC' pour quitter")
    print("📱 Positionnez votre visage face à la caméra")
    print()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Erreur de lecture de la caméra")
                break
            
            # Traiter la frame
            processed_frame = detector.process_frame(frame)
            
            # Afficher la frame
            cv2.imshow('Heart Rate Detector - OpenCV', processed_frame)
            
            # Gestion des touches
            key = cv2.waitKey(1) & 0xFF
            if key != 255:  # Une touche a été pressée
                if not detector.handle_key(key):
                    break
    
    except KeyboardInterrupt:
        print("\n🛑 Arrêt demandé par l'utilisateur")
    
    except Exception as e:
        print(f"❌ Erreur inattendue : {e}")
    
    finally:
        # Libérer les ressources
        cap.release()
        cv2.destroyAllWindows()
        print("🏁 Arrêt propre du programme")


if __name__ == "__main__":
    main()
