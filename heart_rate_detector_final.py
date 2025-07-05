"""
Détecteur de rythme card        bpm_color = self.get_bpm_color(self.current_bpm)
        bmp_text = f"BPM: {self.current_bpm:.2f}" if self.current_bpm > 0 else "BPM: Calcul..."
        cv2.putText(frame, bpm_text, (15, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, bpm_color, 3)e OPTIMISÉ - VERSION ULTRA RAPIDE
===========================================================

Version optimisée avec :
✅ BPM ultra-granulaire et lissé (pas de sauts brusques)
✅ Graphique BPM fluide avec plus de points
✅ Styles UI vraiment différents et effectifs
✅ Notifications avec styles visuels distincts
✅ Performances optimisées (latence réduite)???
✅ Calcul BPM plus précis avec interpolation
✅ Inter        # BPM géant au centre
        bpm_color = self.get_bpm_color(self.current_bpm)
        bpm_text = f"{self.current_bpm:.2f}" if self.current_bpm > 0 else "--"
        
        # Taille de police adaptative
        font_scale = min(w / 400, h / 300)
        cv2.putText(frame, bpm_text, (w // 2 - 80, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale * 2, bpm_color, 4)  # Graphique BPM fluide
        if self.show_bpm_graph and len(self.detailed_bpm_history) > 10: réactive et stable

Optimisations :
- Calc        b        # Texte BPM
        cv2.putText(frame, bpm_text, (15, 35), 
                   cv2.FONT_        bmp_text = f"{self.current_bpm:.2f}" if self.current_bpm > 0 else "--"
        
        # Taille de police adaptative
        font_scale = min(w / 400, h / 300)
        cv2.putText(frame, bmp_text, (w // 2 - 80, 70),   bmp_text = f"{self.current_bpm:.2f}" if self.current_bpm > 0 else "--"ERSHEY_SIMPLEX, 0.9, bpm_color, 2)color = self.get_bpm_color(self.current_bpm)
        bpm_text = f"BPM: {self.current_bpm:.1f}" if self        print("🔄 'R' : Reset")
        print("🖥️ 'F11' ou 'Shift+F' : Basculer plein écran")
        print("🚪 'ESC' ou 'Q' : Quitter")
        print()
        print("💡 ASTUCE : La fenêtre est redimensionnable - tirez les bords pour l'agrandir !")
        print("🖼️ CONSEIL : Utilisez F11 ou Shift+F pour passer en plein écran")
        print()nt_bpm > 0 else "BPM: --" BPM avec interpolation        # BPM principal
        bpm_color = self.get_bpm_color(self.current_bpm)
        bpm_text = f"BPM: {self.current_bpm:.1f}" if self.current_bpm > 0 else "BPM: Calcul..."
        cv2.putText(frame, bpm_text, (15, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, bpm_color, 3) plus de fluidité
- Graphique avec plus de points et lissage
- Styles UI : minimal, complet, plein écran (vraiment différents)
- Notifications : classique, moderne, minimal (styles visuels)
- Cache pour les calculs lourds
- Réduction des appels coûteux
- Amélioration de la fréquence d'affichage

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
"""

import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from collections import deque
import time
import sys
import os
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d

# Ajouter le dossier courant au path pour importer nos modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import SignalProcessor
from utils_manual import ManualROISelector, HeartRateVisualizer
from vascular_analyzer_simple import VascularMicroMovementAnalyzer
import config


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
        
        # Convertir en numpy array
        signal_array = np.array(list(self.signal_buffer))
        
        # Appliquer les filtres coupe-bande
        filtered_signal = signal_array.copy()
        for name, (b, a) in self.notch_filters:
            try:
                filtered_signal = filtfilt(b, a, filtered_signal)
            except:
                pass
        
        # Calculer le BPM brut
        raw_bpm = self.signal_processor.calculate_bpm_from_fft(filtered_signal)
        
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


class OptimizedHeartRateDetector:
    """
    Détecteur de rythme cardiaque optimisé avec toutes les améliorations
    """
    
    def __init__(self):
        # Initialisation des composants
        self.signal_processor = OptimizedSignalProcessor(fps=30)
        self.manual_roi_selector = ManualROISelector()
        self.heart_rate_visualizer = HeartRateVisualizer()
        self.vascular_analyzer = VascularMicroMovementAnalyzer(
            history_size=5, sensitivity=0.1  # Réduit pour les performances
        )
        
        # MediaPipe pour la détection du visage
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
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
        
    def detect_forehead_roi(self, frame):
        """Détecte automatiquement la ROI du front (avec cache)"""
        current_time = time.time()
        
        # Utiliser le cache si récent (optimisation)
        if current_time - self.last_roi_time < 0.5:  # 500ms de cache
            return self.roi_cache.get('last_roi', None)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            self.face_detection_count += 1
            face_landmarks = results.multi_face_landmarks[0]
            h, w, _ = frame.shape
            
            # Points du front améliorés
            forehead_indices = [10, 151, 9, 175, 150, 136, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323]
            
            # Calculer la bounding box du front
            x_coords = [int(face_landmarks.landmark[i].x * w) for i in forehead_indices]
            y_coords = [int(face_landmarks.landmark[i].y * h) for i in forehead_indices]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # ROI optimisée
            width = x_max - x_min
            height = y_max - y_min
            
            # Centrer sur le front avec une taille optimale
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            
            # Taille optimale pour rPPG
            optimal_size = min(width, height, 120)
            
            roi_x = max(0, center_x - optimal_size // 2)
            roi_y = max(0, center_y - optimal_size // 2)
            roi_w = min(optimal_size, w - roi_x)
            roi_h = min(optimal_size, h - roi_y)
            
            roi = (roi_x, roi_y, roi_w, roi_h)
            
            # Mise à jour du cache
            self.roi_cache['last_roi'] = roi
            self.last_roi_time = current_time
            
            return roi
        
        return None
    
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
        
    def toggle_vascular_heatmap(self):
        """Bascule la heatmap vasculaire"""
        self.show_vascular_heatmap = not self.show_vascular_heatmap
        status = "activée" if self.show_vascular_heatmap else "désactivée"
        self.add_notification(f"Heatmap {status}", "info")
        
    def toggle_vascular_hotspots(self):
        """Bascule les hotspots vasculaires"""
        self.show_vascular_hotspots = not self.show_vascular_hotspots
        status = "activés" if self.show_vascular_hotspots else "désactivés"
        self.add_notification(f"Hotspots {status}", "info")
        
    def toggle_pulse_overlay(self):
        """Bascule la surcouche pulsante"""
        self.show_pulse_overlay = not self.show_pulse_overlay
        status = "activée" if self.show_pulse_overlay else "désactivée"
        self.add_notification(f"Surcouche pulsante {status}", "info")
        
    def toggle_bpm_graph(self):
        """Bascule le graphique BPM"""
        self.show_bpm_graph = not self.show_bpm_graph
        status = "activé" if self.show_bpm_graph else "désactivé"
        self.add_notification(f"Graphique BPM {status}", "info")
        
    def toggle_ui_style(self):
        """Change vraiment le style d'interface"""
        current_idx = self.ui_styles.index(self.ui_style)
        self.ui_style = self.ui_styles[(current_idx + 1) % len(self.ui_styles)]
        self.add_notification(f"Style UI: {self.ui_style.upper()}", "info")
        
    def toggle_notification_style(self):
        """Change vraiment le style de notification"""
        current_idx = self.notification_styles.index(self.notification_style)
        self.notification_style = self.notification_styles[(current_idx + 1) % len(self.notification_styles)]
        self.add_notification(f"Style notifications: {self.notification_style.upper()}", "info")
    
    def process_frame(self, frame):
        """Traite une frame avec optimisations"""
        if frame is None:
            return None
        
        self.frame_count += 1
        
        # Gestion de la sélection de ROI
        if self.mode == "auto":
            detected_roi = self.detect_forehead_roi(frame)
            if detected_roi:
                self.current_roi = detected_roi
        elif self.mode == "manual":
            if self.is_selecting_roi:
                selected_roi = self.manual_roi_selector.get_roi()
                if selected_roi:
                    self.current_roi = selected_roi
                    self.is_selecting_roi = False
                    self.add_notification("ROI sélectionnée", "success")
        
        # Analyse vasculaire (seulement si nécessaire)
        if self.show_vascular_heatmap or self.show_vascular_hotspots:
            self.vascular_analyzer.update_frame(frame)
            
            # Heatmap vasculaire
            if self.show_vascular_heatmap:
                frame = self.vascular_analyzer.draw_movement_overlay(frame)
            
            # Hotspots vasculaires
            if self.show_vascular_hotspots:
                frame = self.vascular_analyzer.draw_hotspots(frame)
        
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
        
        # Graphique BPM fluide
        if self.show_bpm_graph and len(self.detailed_bpm_history) > 10:
            graph_img = self.create_smooth_bpm_graph()
            if graph_img is not None:
                frame = self.overlay_graph(frame, graph_img)
        
        # Interface utilisateur (selon le style choisi)
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
    
    def create_smooth_bpm_graph(self):
        """Crée un graphique BPM ultra-fluide"""
        try:
            if len(self.detailed_bpm_history) < 10:
                return None
            
            # Configuration du graphique
            fig, ax = plt.subplots(figsize=(4, 2), facecolor='black')
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')
            
            # Préparer les données avec interpolation
            x = np.array(list(range(len(self.detailed_bpm_history))))
            y = np.array(list(self.detailed_bpm_history))
            
            # Interpolation pour plus de fluidité
            if len(x) > 3:
                x_smooth = np.linspace(x.min(), x.max(), len(x) * 3)
                f = interp1d(x, y, kind='cubic', fill_value='extrapolate')
                y_smooth = f(x_smooth)
            else:
                x_smooth = x
                y_smooth = y
            
            # Graphique principal
            ax.plot(x_smooth, y_smooth, 'red', linewidth=2, alpha=0.9)
            ax.fill_between(x_smooth, y_smooth, alpha=0.2, color='red')
            
            # Points de données réels
            ax.scatter(x, y, c='yellow', s=20, alpha=0.7, zorder=5)
            
            # Ligne de moyenne mobile
            if len(y) > 10:
                window_size = min(10, len(y))
                avg_y = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
                avg_x = x[window_size-1:]
                ax.plot(avg_x, avg_y, 'yellow', linestyle='--', alpha=0.8, linewidth=1)
            
            # Configuration des axes
            ax.set_ylim(40, 180)
            ax.set_xlim(0, len(x_smooth))
            ax.set_title('BPM Temps Réel', color='white', fontsize=8)
            ax.set_xlabel('Temps', color='white', fontsize=6)
            ax.set_ylabel('BPM', color='white', fontsize=6)
            ax.tick_params(colors='white', labelsize=5)
            ax.grid(True, alpha=0.2, color='white')
            
            # Zones de BPM
            ax.axhspan(50, 90, alpha=0.1, color='green')
            ax.axhspan(90, 150, alpha=0.1, color='yellow')
            
            # Convertir en image
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            renderer = canvas.get_renderer()
            raw_data = renderer.tostring_rgb()
            size = canvas.get_width_height()
            
            img = np.frombuffer(raw_data, dtype=np.uint8)
            img = img.reshape(size[1], size[0], 3)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            plt.close(fig)
            return img
            
        except Exception as e:
            return None
    
    def overlay_graph(self, frame, graph_img):
        """Superpose le graphique sur la frame"""
        try:
            if frame is None or graph_img is None:
                return frame
            
            h, w = frame.shape[:2]
            gh, gw = graph_img.shape[:2]
            
            # Position selon le style UI
            if self.ui_style == "minimal":
                x, y = 10, h - gh - 10
            elif self.ui_style == "fullscreen":
                x, y = (w - gw) // 2, h - gh - 10
            else:  # complete
                x, y = w - gw - 10, 150
            
            # Vérifier les limites
            if x + gw <= w and y + gh <= h:
                # Fond semi-transparent
                overlay = frame.copy()
                cv2.rectangle(overlay, (x-5, y-5), (x+gw+5, y+gh+5), (0, 0, 0), -1)
                frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
                
                # Superposer le graphique
                frame[y:y+gh, x:x+gw] = graph_img
            
            return frame
            
        except Exception as e:
            return frame
    
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
        bpm_text = f"BPM: {self.current_bpm:.2f}" if self.current_bpm > 0 else "BPM: --"
        
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
        bpm_text = f"BPM: {self.current_bpm:.2f}" if self.current_bpm > 0 else "BPM: Calcul..."
        cv2.putText(frame, bpm_text, (15, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, bpm_color, 3)
        
        # Statistiques
        if self.avg_bpm_history:
            avg_bpm = np.mean(list(self.avg_bpm_history))
            cv2.putText(frame, f"Moyenne: {avg_bpm:.2f}", (15, 85), 
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
        cv2.rectangle(overlay, (w - 220, 10), (w - 10, 300), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        cv2.rectangle(frame, (w - 220, 10), (w - 10, 300), (100, 100, 100), 2)
        
        # Contrôles
        cv2.putText(frame, "CONTROLES", (w - 215, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['accent'], 1)
        
        controls = [
            "A: Mode auto",
            "M: Mode manuel", 
            "V: Heatmap",
            "H: Hotspots",
            "P: Pulse",
            "G: Graphique",
            "F: Style UI",
            "N: Notifications",
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
        bpm_text = f"{self.current_bpm:.2f}" if self.current_bpm > 0 else "--"
        
        # Taille de police adaptative
        font_scale = min(w / 400, h / 300)
        cv2.putText(frame, bpm_text, (w // 2 - 80, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale * 2, bpm_color, 4)
        
        # Label BPM
        cv2.putText(frame, "BPM", (w // 2 - 30, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['text'], 2)
        
        # Statistiques étendues sur les côtés
        if self.avg_bpm_history:
            avg_bpm = np.mean(list(self.avg_bpm_history))
            cv2.putText(frame, f"Moyenne: {avg_bpm:.2f}", (50, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 1)
        
        if len(self.bpm_history) > 1:
            trend = self.bpm_history[-1] - self.bpm_history[-2]
            trend_text = f"Tendance: {trend:+.2f}"
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
                x, y = 10, 60 + i * 25
                self.draw_minimal_notification(frame, message, msg_type, x, y, age)
            elif self.notification_style == "modern":
                x, y = w - 300, 60 + i * 30
                self.draw_modern_notification(frame, message, msg_type, x, y, age)
            else:  # classic
                x, y = w // 2 - 150, 60 + i * 25
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
    
    def run(self):
        """Lance l'application principale"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Impossible d'accéder à la caméra")
            return
        
        # Configuration de la caméra pour de meilleures performances
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Créer la fenêtre avec des propriétés redimensionnables
        window_name = '💗 Heart Rate Monitor - OPTIMISÉ'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)  # Taille par défaut
        
        print("🚀 === DÉTECTEUR DE RYTHME CARDIAQUE OPTIMISÉ ===")
        print("✨ Version ultra-rapide avec BPM granulaire")
        print()
        print("🎯 CONTRÔLES :")
        print("🔄 'A' : Mode automatique")
        print("✋ 'M' : Mode manuel")
        print("🌡️ 'V' : Heatmap vasculaire")
        print("🔥 'H' : Hotspots")
        print("💗 'P' : Surcouche pulsante")
        print("📈 'G' : Graphique BPM")
        print("🎨 'F' : Style UI (minimal/complet/plein écran)")
        print("📢 'N' : Style notifications")
        print("🔄 'R' : Reset")
        print("�️ 'F11' : Basculer plein écran")
        print("�🚪 'ESC' ou 'Q' : Quitter")
        print()
        print("💡 ASTUCE : La fenêtre est redimensionnable - tirez les bords !")
        print()
        
        self.add_notification("Détecteur optimisé lancé!", "success")
        
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
                elif key == ord('v'):
                    self.toggle_vascular_heatmap()
                elif key == ord('h'):
                    self.toggle_vascular_hotspots()
                elif key == ord('p'):
                    self.toggle_pulse_overlay()
                elif key == ord('g'):
                    self.toggle_bpm_graph()
                elif key == ord('f'):
                    self.toggle_ui_style()
                elif key == ord('n'):
                    self.toggle_notification_style()
                elif key == ord('r'):
                    self.reset_all()
                elif key == 225 or key == ord('F'):  # F11 ou Shift+F pour plein écran
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
            plt.close('all')
            
            # Statistiques finales
            elapsed_time = time.time() - self.start_time
            fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            
            print(f"\n📊 === STATISTIQUES FINALES ===")
            print(f"⏱️ Durée: {elapsed_time:.1f}s")
            print(f"🎬 Frames: {self.frame_count}")
            print(f"📺 FPS moyen: {fps:.1f}")
            print(f"👤 Faces détectées: {self.face_detection_count}")
            if self.avg_bpm_history:
                print(f"💓 BPM moyen: {np.mean(list(self.avg_bpm_history)):.2f}")
                print(f"💗 BPM final: {self.current_bpm:.2f}")
            if self.signal_quality_history:
                print(f"📈 Qualité moyenne: {np.mean(list(self.signal_quality_history)):.1f}%")
            print("✅ Session terminée avec succès!")
    
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
        self.bpm_history.clear()
        self.detailed_bpm_history.clear()
        self.signal_quality_history.clear()
        
        # Réinitialiser les métriques
        self.current_bpm = 0
        self.current_roi = None
        
        # Réinitialiser l'analyseur vasculaire
        if hasattr(self.vascular_analyzer, 'reset'):
            self.vascular_analyzer.reset()
        
        # Réinitialiser les caches
        self.roi_cache.clear()
        
        self.add_notification("Système réinitialisé!", "success")


def main():
    """Point d'entrée principal"""
    try:
        detector = OptimizedHeartRateDetector()
        detector.run()
    except Exception as e:
        print(f"❌ Erreur fatale: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
