"""
Détecteur de rythme cardiaque OPTIMISÉ - VERSION OPENCV PURE
===========================================================

Version compatible Python 3.13 utilisant OpenCV DNN au lieu de MediaPipe
✅ Fenêtre redimensionnable et mode plein écran
✅ Détection de visage avec OpenCV DNN
✅ BPM ultra-granulaire et lissé
✅ Interface scalable
"""

import cv2
import numpy as np
from collections import deque
import time
import sys
import os    def draw_minimal_ui(self, frame):
        """Interface minimale - juste le BPM"""
        bpm_color = self.get_bpm_color(self.current_bpm)
        bpm_text = f"BPM: {self.current_bpm:.1f}" if self.current_bpm > 0 else "BPM: --"
        
        # Fond minimal
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (200, 50), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
        
        # Texte BPM
        cv2.putText(frame, bmp_text, (15, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, bmp_color, 2)
        
        return framerequest

# Ajouter le dossier courant au path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils import SignalProcessor
    from utils_manual import ManualROISelector, HeartRateVisualizer
    from vascular_analyzer_simple import VascularMicroMovementAnalyzer
except ImportError:
    print("⚠️ Modules utils non trouvés - Utilisation de versions simplifiées")
    
    class SignalProcessor:
        def __init__(self, fps=30):
            self.fps = fps
            
        def calculate_bpm_from_fft(self, signal):
            """Calcul BPM simple via FFT"""
            if len(signal) < 60:
                return 0
            
            # FFT simple
            fft = np.fft.fft(signal)
            freqs = np.fft.fftfreq(len(signal), 1/self.fps)
            
            # Filtrer les fréquences cardiaques (0.5-4 Hz = 30-240 BPM)
            valid_freqs = (freqs >= 0.5) & (freqs <= 4.0)
            if not np.any(valid_freqs):
                return 0
                
            # Trouver le pic dominant
            power = np.abs(fft[valid_freqs])
            peak_idx = np.argmax(power)
            peak_freq = freqs[valid_freqs][peak_idx]
            
            return peak_freq * 60  # Convertir en BPM
    
    class ManualROISelector:
        def __init__(self):
            self.roi = None
            self.selecting = False
            self.start_point = None
            self.end_point = None
            
        def mouse_callback(self, event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.selecting = True
                self.start_point = (x, y)
            elif event == cv2.EVENT_LBUTTONUP and self.selecting:
                self.selecting = False
                self.end_point = (x, y)
                if self.start_point and self.end_point:
                    x1, y1 = self.start_point
                    x2, y2 = self.end_point
                    roi_x = min(x1, x2)
                    roi_y = min(y1, y2)
                    roi_w = abs(x2 - x1)
                    roi_h = abs(y2 - y1)
                    if roi_w >= 20 and roi_h >= 20:
                        self.roi = (roi_x, roi_y, roi_w, roi_h)
        
        def get_roi(self):
            return self.roi
    
    class HeartRateVisualizer:
        def draw_pulse_overlay(self, frame, roi, current_bpm=0):
            return frame
    
    class VascularMicroMovementAnalyzer:
        def __init__(self, **kwargs):
            pass
        def update_frame(self, frame):
            pass
        def draw_movement_overlay(self, frame):
            return frame
        def draw_hotspots(self, frame):
            return frame


class OpenCVFaceDetector:
    """Détecteur de visage utilisant OpenCV DNN"""
    
    def __init__(self):
        self.net = None
        self.confidence_threshold = 0.5
        self.setup_face_detection()
    
    def setup_face_detection(self):
        """Configure la détection de visage avec OpenCV DNN"""
        try:
            # Utiliser le détecteur de visage Haar Cascade intégré
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            print("✅ Détecteur de visage OpenCV initialisé")
        except Exception as e:
            print(f"⚠️ Erreur initialisation détecteur: {e}")
            self.face_cascade = None
    
    def detect_face_roi(self, frame):
        """Détecte le visage et retourne une ROI pour le front"""
        if self.face_cascade is None:
            return None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Prendre le plus grand visage
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            # Extraire la région du front (tiers supérieur du visage)
            forehead_y = y + int(h * 0.1)  # 10% du haut
            forehead_h = int(h * 0.3)      # 30% de la hauteur
            forehead_x = x + int(w * 0.25) # 25% des côtés
            forehead_w = int(w * 0.5)      # 50% de la largeur
            
            # Limiter à 80x80 pixels max pour de meilleures performances
            max_size = 80
            if forehead_w > max_size or forehead_h > max_size:
                center_x = forehead_x + forehead_w // 2
                center_y = forehead_y + forehead_h // 2
                forehead_x = center_x - max_size // 2
                forehead_y = center_y - max_size // 2
                forehead_w = forehead_h = max_size
            
            # Vérifier les limites
            h_frame, w_frame = frame.shape[:2]
            forehead_x = max(0, forehead_x)
            forehead_y = max(0, forehead_y)
            forehead_w = min(forehead_w, w_frame - forehead_x)
            forehead_h = min(forehead_h, h_frame - forehead_y)
            
            if forehead_w > 20 and forehead_h > 20:
                return (forehead_x, forehead_y, forehead_w, forehead_h)
        
        return None


class OptimizedSignalProcessor:
    """Processeur de signal optimisé avec BPM ultra-granulaire"""
    
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
        
        # Convertir en numpy array
        signal_array = np.array(list(self.signal_buffer))
        
        # Calculer le BPM brut
        raw_bpm = self.signal_processor.calculate_bpm_from_fft(signal_array)
        
        # Validation du BPM
        if raw_bpm < 30 or raw_bpm > 200:
            return self.cached_bpm
        
        # Ajouter au buffer de BPM bruts
        self.raw_bpm_buffer.append(raw_bpm)
        
        # Calculer le BPM lissé
        if len(self.raw_bpm_buffer) >= 3:
            # Moyenner avec pondération (plus récent = plus important)
            weights = np.exp(np.linspace(-1, 0, len(self.raw_bpm_buffer)))
            weighted_avg = np.average(list(self.raw_bpm_buffer), weights=weights)
            
            # Interpolation pour plus de fluidité
            self.bpm_buffer.append(weighted_avg)
            if len(self.bpm_buffer) >= 3:
                smooth_bpm = np.median(list(self.bpm_buffer))
            else:
                smooth_bpm = weighted_avg
        else:
            smooth_bpm = raw_bpm
        
        # Mise à jour du cache
        self.cached_bpm = smooth_bpm
        self.last_bpm_time = current_time
        
        return smooth_bpm


class HeartRateDetectorOpenCV:
    """Détecteur de rythme cardiaque avec OpenCV pur - Compatible Python 3.13"""
    
    def __init__(self):
        # Initialisation des composants
        self.signal_processor = OptimizedSignalProcessor(fps=30)
        self.manual_roi_selector = ManualROISelector()
        self.heart_rate_visualizer = HeartRateVisualizer()
        self.vascular_analyzer = VascularMicroMovementAnalyzer()
        self.face_detector = OpenCVFaceDetector()
        
        # Variables d'état
        self.mode = "auto"  # "auto" ou "manual"
        self.current_roi = None
        self.is_selecting_roi = False
        self.is_fullscreen = False
        
        # Métriques et historique
        self.current_bpm = 0
        self.bpm_history = deque(maxlen=100)
        self.detailed_bpm_history = deque(maxlen=300)
        self.avg_bpm_history = deque(maxlen=50)
        self.signal_quality_history = deque(maxlen=30)
        
        # Compteurs
        self.frame_count = 0
        self.face_detection_count = 0
        self.start_time = time.time()
        
        # Options d'affichage
        self.show_vascular_heatmap = False
        self.show_vascular_hotspots = False
        self.show_pulse_overlay = True
        self.show_bpm_graph = False  # Désactivé par défaut pour éviter matplotlib
        
        # Styles UI
        self.ui_styles = ["minimal", "complete", "fullscreen"]
        self.ui_style = "complete"
        
        # Styles de notification
        self.notification_styles = ["classic", "modern", "minimal"]
        self.notification_style = "modern"
        
        # Variables pour les notifications
        self.notification_queue = deque(maxlen=5)
        
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
        
    def add_notification(self, message, type="info"):
        """Ajoute une notification à la queue"""
        self.notification_queue.append({
            'message': message,
            'type': type,
            'time': time.time()
        })
        
    def get_bpm_color(self, bpm):
        """Retourne la couleur selon le BPM"""
        if bpm == 0:
            return self.colors['inactive']
        elif bpm < self.bpm_thresholds['danger_min'] or bpm > self.bpm_thresholds['danger_max']:
            return self.colors['danger']
        elif self.bpm_thresholds['rest_min'] <= bpm <= self.bpm_thresholds['rest_max']:
            return self.colors['good']
        elif self.bpm_thresholds['active_min'] <= bpm <= self.bpm_thresholds['active_max']:
            return self.colors['warning']
        else:
            return self.colors['accent']
    
    def calculate_signal_quality(self, roi_frame):
        """Calcule la qualité du signal"""
        if roi_frame.size == 0:
            return 0
        
        # Calculer la variance du canal vert (indicateur de qualité)
        green_channel = roi_frame[:, :, 1]
        mean_val = np.mean(green_channel)
        var_val = np.var(green_channel)
        
        # Normaliser entre 0 et 100
        quality = min(100, max(0, (var_val / mean_val) * 100 if mean_val > 0 else 0))
        return quality
    
    def switch_to_automatic_mode(self):
        """Passe en mode automatique"""
        self.mode = "auto"
        self.is_selecting_roi = False
        self.add_notification("Mode automatique activé", "info")
        
    def switch_to_manual_mode(self):
        """Passe en mode manuel"""
        self.mode = "manual"
        self.is_selecting_roi = True
        self.add_notification("Mode manuel - Sélectionnez la zone", "info")
        
    def toggle_pulse_overlay(self):
        """Bascule la surcouche pulsante"""
        self.show_pulse_overlay = not self.show_pulse_overlay
        status = "activée" if self.show_pulse_overlay else "désactivée"
        self.add_notification(f"Surcouche pulsante {status}", "info")
        
    def toggle_ui_style(self):
        """Change le style d'interface"""
        current_idx = self.ui_styles.index(self.ui_style)
        self.ui_style = self.ui_styles[(current_idx + 1) % len(self.ui_styles)]
        self.add_notification(f"Style UI: {self.ui_style.upper()}", "info")
        
    def toggle_notification_style(self):
        """Change le style de notification"""
        current_idx = self.notification_styles.index(self.notification_style)
        self.notification_style = self.notification_styles[(current_idx + 1) % len(self.notification_styles)]
        self.add_notification(f"Style notifications: {self.notification_style.upper()}", "info")
    
    def process_frame(self, frame):
        """Traite une frame"""
        if frame is None:
            return None
        
        self.frame_count += 1
        
        # Gestion de la sélection de ROI
        if self.mode == "auto":
            detected_roi = self.face_detector.detect_face_roi(frame)
            if detected_roi:
                self.current_roi = detected_roi
                self.face_detection_count += 1
        elif self.mode == "manual":
            if self.is_selecting_roi:
                selected_roi = self.manual_roi_selector.get_roi()
                if selected_roi:
                    self.current_roi = selected_roi
                    self.is_selecting_roi = False
                    self.add_notification("ROI sélectionnée", "success")
        
        # Traitement du signal rPPG
        signal_quality = 0
        if self.current_roi:
            x, y, w, h = self.current_roi
            
            # Extraire le signal rPPG
            roi_frame = frame[y:y+h, x:x+w]
            if roi_frame.size > 0:
                # Calculer la qualité du signal
                signal_quality = self.calculate_signal_quality(roi_frame)
                self.signal_quality_history.append(signal_quality)
                
                # Extraire le signal de la ROI (canal vert)
                signal = np.mean(roi_frame[:, :, 1])
                
                # Traiter le signal
                if signal is not None:
                    self.signal_processor.add_signal(signal)
                    new_bpm = self.signal_processor.get_smooth_bpm()
                    
                    if new_bpm > 0:
                        self.current_bpm = new_bpm
                        self.bpm_history.append(new_bpm)
                        self.detailed_bpm_history.append(new_bpm)
                        self.avg_bpm_history.append(new_bpm)
                
                # Dessiner la ROI
                self.draw_roi(frame, (x, y, w, h), signal_quality)
                
                # Surcouche pulsante
                if self.show_pulse_overlay and self.current_bpm > 0:
                    frame = self.heart_rate_visualizer.draw_pulse_overlay(
                        frame, (x, y, w, h), self.current_bpm
                    )
        
        # Interface utilisateur
        frame = self.draw_ui_with_style(frame, signal_quality)
        
        # Notifications
        frame = self.draw_notifications(frame)
        
        return frame
    
    def draw_roi(self, frame, roi, signal_quality):
        """Dessine la ROI avec indicateurs de qualité"""
        x, y, w, h = roi
        
        # Couleur selon la qualité
        if signal_quality > 70:
            color = self.colors['good']
        elif signal_quality > 40:
            color = self.colors['warning']
        else:
            color = self.colors['danger']
        
        # Rectangle principal
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Indicateur de qualité
        cv2.putText(frame, f"Q: {signal_quality:.0f}%", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def draw_ui_with_style(self, frame, signal_quality):
        """Dessine l'interface selon le style choisi"""
        if self.ui_style == "minimal":
            return self.draw_minimal_ui(frame)
        elif self.ui_style == "fullscreen":
            return self.draw_fullscreen_ui(frame, signal_quality)
        else:  # complete
            return self.draw_complete_ui(frame, signal_quality)
    
    def draw_minimal_ui(self, frame):
        """Interface minimale - juste le BPM"""
        bpm_color = self.get_bpm_color(self.current_bpm)
        bmp_text = f"BPM: {self.current_bpm:.1f}" if self.current_bpm > 0 else "BPM: --"
        
        # Fond minimal
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (200, 50), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
        
        # Texte BPM
        cv2.putText(frame, bpm_text, (15, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, bpm_color, 2)
        
        return frame
    
    def draw_complete_ui(self, frame, signal_quality):
        """Interface complète avec tous les panneaux"""
        h, w = frame.shape[:2]
        
        # Panel principal gauche
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (320, 140), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        cv2.rectangle(frame, (10, 10), (320, 140), (100, 100, 100), 2)
        
        # Titre
        cv2.putText(frame, "HEART RATE MONITOR", (15, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['accent'], 2)
        
        # BPM principal
        bpm_color = self.get_bpm_color(self.current_bpm)
        bmp_text = f"BPM: {self.current_bpm:.1f}" if self.current_bpm > 0 else "BPM: Calcul..."
        cv2.putText(frame, bmp_text, (15, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, bpm_color, 3)
        
        # Statistiques
        if self.avg_bpm_history:
            avg_bpm = np.mean(list(self.avg_bpm_history))
            cv2.putText(frame, f"Moyenne: {avg_bpm:.1f}", (15, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        if self.signal_quality_history:
            avg_quality = np.mean(list(self.signal_quality_history))
            cv2.putText(frame, f"Qualite: {avg_quality:.0f}%", (15, 105), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        # Mode et statut
        cv2.putText(frame, f"Mode: {self.mode.upper()}", (15, 125), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['accent'], 1)
        
        # Panel de contrôles droite
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - 220, 10), (w - 10, 250), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        cv2.rectangle(frame, (w - 220, 10), (w - 10, 250), (100, 100, 100), 2)
        
        # Contrôles
        cv2.putText(frame, "CONTROLES", (w - 215, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['accent'], 1)
        
        controls = [
            "A: Mode auto",
            "M: Mode manuel", 
            "P: Pulse",
            "F: Style UI",
            "N: Notifications",
            "F11: Plein ecran",
            "R: Reset",
            "Q: Quitter"
        ]
        
        for i, control in enumerate(controls):
            cv2.putText(frame, control, (w - 210, 50 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.colors['text'], 1)
        
        return frame
    
    def draw_fullscreen_ui(self, frame, signal_quality):
        """Interface plein écran - informations maximales"""
        h, w = frame.shape[:2]
        
        # Fond semi-transparent sur toute la hauteur
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
        
        # BPM géant au centre
        bpm_color = self.get_bpm_color(self.current_bpm)
        bmp_text = f"{self.current_bpm:.1f}" if self.current_bpm > 0 else "--"
        
        # Taille de police adaptative
        font_scale = min(w / 400, h / 300)
        cv2.putText(frame, bmp_text, (w // 2 - 80, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale * 2, bpm_color, 4)
        
        # Label BPM
        cv2.putText(frame, "BPM", (w // 2 - 30, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['text'], 2)
        
        # Statistiques étendues sur les côtés
        if self.avg_bpm_history:
            avg_bpm = np.mean(list(self.avg_bpm_history))
            cv2.putText(frame, f"Moyenne: {avg_bpm:.1f}", (50, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 1)
        
        if len(self.bpm_history) > 1:
            trend = self.bpm_history[-1] - self.bmp_history[-2]
            trend_text = f"Tendance: {trend:+.1f}"
            trend_color = self.colors['good'] if trend >= 0 else self.colors['danger']
            cv2.putText(frame, trend_text, (w - 200, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, trend_color, 1)
        
        return frame
    
    def draw_notifications(self, frame):
        """Dessine les notifications selon le style choisi"""
        if not self.notification_queue:
            return frame
        
        current_time = time.time()
        h, w = frame.shape[:2]
        
        # Nettoyer les notifications expirées
        self.notification_queue = deque([
            n for n in self.notification_queue 
            if current_time - n['time'] < 3.0
        ], maxlen=5)
        
        # Dessiner les notifications
        for i, notification in enumerate(self.notification_queue):
            message = notification['message']
            msg_type = notification['type']
            age = current_time - notification['time']
            
            # Position selon le style
            if self.notification_style == "minimal":
                x, y = 10, 160 + i * 25
                self.draw_minimal_notification(frame, message, msg_type, x, y, age)
            elif self.notification_style == "modern":
                x, y = w - 300, 160 + i * 30
                self.draw_modern_notification(frame, message, msg_type, x, y, age)
            else:  # classic
                x, y = w // 2 - 150, 160 + i * 25
                self.draw_classic_notification(frame, message, msg_type, x, y, age)
        
        return frame
    
    def draw_minimal_notification(self, frame, message, msg_type, x, y, age):
        """Notification minimale - juste le texte"""
        alpha = max(0.3, 1.0 - age / 3.0)
        
        # Couleur selon le type
        if msg_type == "success":
            color = self.colors['good']
        elif msg_type == "warning":
            color = self.colors['warning']
        elif msg_type == "error":
            color = self.colors['danger']
        else:
            color = self.colors['text']
        
        # Appliquer l'alpha
        color = tuple(int(c * alpha) for c in color)
        
        cv2.putText(frame, message, (x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def draw_modern_notification(self, frame, message, msg_type, x, y, age):
        """Notification moderne avec fond et bordure"""
        alpha = max(0.3, 1.0 - age / 3.0)
        
        # Couleur selon le type
        if msg_type == "success":
            bg_color = (0, 100, 0)
            border_color = (0, 255, 0)
        elif msg_type == "warning":
            bg_color = (100, 100, 0)
            border_color = (0, 255, 255)
        elif msg_type == "error":
            bg_color = (100, 0, 0)
            border_color = (0, 0, 255)
        else:
            bg_color = (50, 50, 50)
            border_color = (150, 150, 150)
        
        # Fond avec transparence
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 5, y - 20), (x + 250, y + 5), bg_color, -1)
        cv2.rectangle(overlay, (x - 5, y - 20), (x + 250, y + 5), border_color, 2)
        frame = cv2.addWeighted(frame, 1 - alpha * 0.7, overlay, alpha * 0.7, 0)
        
        # Texte
        cv2.putText(frame, message, (x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def draw_classic_notification(self, frame, message, msg_type, x, y, age):
        """Notification classique avec fond simple"""
        alpha = max(0.3, 1.0 - age / 3.0)
        
        # Fond simple
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 5, y - 15), (x + 300, y + 5), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 1 - alpha * 0.5, overlay, alpha * 0.5, 0)
        
        # Texte
        cv2.putText(frame, message, (x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def toggle_fullscreen(self, window_name):
        """Bascule entre le mode plein écran et le mode fenêtré"""
        if self.is_fullscreen:
            # Revenir au mode fenêtré
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 800, 600)
            self.is_fullscreen = False
            self.add_notification("Mode fenêtré activé", "info")
        else:
            # Passer en plein écran
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            self.is_fullscreen = True
            self.add_notification("Mode plein écran activé", "info")
    
    def reset_all(self):
        """Réinitialise tout le système"""
        # Réinitialiser les buffers
        self.signal_processor.signal_buffer.clear()
        self.signal_processor.bpm_buffer.clear()
        self.signal_processor.raw_bpm_buffer.clear()
        self.bmp_history.clear()
        self.detailed_bpm_history.clear()
        self.signal_quality_history.clear()
        
        # Réinitialiser les métriques
        self.current_bpm = 0
        self.current_roi = None
        
        self.add_notification("Système réinitialisé!", "success")
    
    def run(self):
        """Lance l'application principale"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Impossible d'accéder à la caméra")
            return
        
        # Configuration de la caméra
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Créer la fenêtre avec des propriétés redimensionnables
        window_name = '💗 Heart Rate Monitor - OpenCV'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
        
        print("🚀 === DÉTECTEUR DE RYTHME CARDIAQUE - OPENCV ===")
        print("✨ Version compatible Python 3.13")
        print()
        print("🎯 CONTRÔLES :")
        print("🔄 'A' : Mode automatique")
        print("✋ 'M' : Mode manuel")
        print("💗 'P' : Surcouche pulsante")
        print("🎨 'F' : Style UI (minimal/complet/plein écran)")
        print("📢 'N' : Style notifications")
        print("🔄 'R' : Reset")
        print("🖥️ 'F11' ou 'Shift+F' : Basculer plein écran")
        print("🚪 'ESC' ou 'Q' : Quitter")
        print()
        print("💡 ASTUCE : La fenêtre est redimensionnable - tirez les bords pour l'agrandir !")
        print("🖼️ CONSEIL : Utilisez F11 ou Shift+F pour passer en plein écran")
        print()
        
        self.add_notification("Détecteur OpenCV lancé!", "success")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Traiter la frame
                processed_frame = self.process_frame(frame)
                
                # Afficher
                if processed_frame is not None:
                    cv2.imshow(window_name, processed_frame)
                else:
                    cv2.imshow(window_name, frame)
                
                # Gestion des événements clavier
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # ESC
                    break
                elif key == ord('a'):
                    self.switch_to_automatic_mode()
                elif key == ord('m'):
                    self.switch_to_manual_mode()
                elif key == ord('p'):
                    self.toggle_pulse_overlay()
                elif key == ord('f'):
                    self.toggle_ui_style()
                elif key == ord('n'):
                    self.toggle_notification_style()
                elif key == ord('r'):
                    self.reset_all()
                elif key == 225 or key == ord('F'):  # F11 ou Shift+F
                    self.toggle_fullscreen(window_name)
                
                # Gestion des événements souris
                if self.mode == "manual" and self.is_selecting_roi:
                    cv2.setMouseCallback(window_name, 
                                       self.manual_roi_selector.mouse_callback)
        
        except KeyboardInterrupt:
            print("\n🛑 Interruption par l'utilisateur")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Statistiques finales
            elapsed_time = time.time() - self.start_time
            fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            
            print(f"\n📊 === STATISTIQUES FINALES ===")
            print(f"⏱️ Durée: {elapsed_time:.1f}s")
            print(f"🎬 Frames: {self.frame_count}")
            print(f"📺 FPS moyen: {fps:.1f}")
            print(f"👤 Faces détectées: {self.face_detection_count}")
            if self.avg_bpm_history:
                print(f"💓 BPM moyen: {np.mean(list(self.avg_bpm_history)):.1f}")
                print(f"💗 BPM final: {self.current_bpm:.1f}")
            if self.signal_quality_history:
                print(f"📈 Qualité moyenne: {np.mean(list(self.signal_quality_history)):.1f}%")
            print("✅ Session terminée avec succès!")


def main():
    """Point d'entrée principal"""
    try:
        detector = HeartRateDetectorOpenCV()
        detector.run()
    except Exception as e:
        print(f"❌ Erreur fatale: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
