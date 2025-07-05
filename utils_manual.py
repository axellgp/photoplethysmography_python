"""
Utilitaires pour le traitement du signal PPG avec sélection manuelle de ROI
"""

import numpy as np
import scipy.signal
from scipy.fft import fft, fftfreq
from collections import deque
import cv2
import time


class ManualROISelector:
    """Classe pour la sélection manuelle de la ROI"""
    
    def __init__(self):
        self.roi = None
        self.selecting = False
        self.start_point = None
        self.end_point = None
        self.mouse_callback_set = False
        
    def mouse_callback(self, event, x, y, flags, param):
        """Callback pour la sélection de ROI à la souris"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selecting = True
            self.start_point = (x, y)
            self.end_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE and self.selecting:
            self.end_point = (x, y)
            
        elif event == cv2.EVENT_LBUTTONUP:
            self.selecting = False
            self.end_point = (x, y)
            
            # Calculer la ROI finale
            if self.start_point and self.end_point:
                x1, y1 = self.start_point
                x2, y2 = self.end_point
                
                # S'assurer que x1,y1 est le coin supérieur gauche
                roi_x = min(x1, x2)
                roi_y = min(y1, y2)
                roi_w = abs(x2 - x1)
                roi_h = abs(y2 - y1)
                
                # Minimum 20x20 pixels
                if roi_w >= 20 and roi_h >= 20:
                    self.roi = (roi_x, roi_y, roi_w, roi_h)
                    print(f"📍 ROI sélectionnée: ({roi_x}, {roi_y}, {roi_w}, {roi_h})")
    
    def setup_mouse_callback(self, window_name):
        """Configure le callback de souris pour une fenêtre"""
        if not self.mouse_callback_set:
            cv2.setMouseCallback(window_name, self.mouse_callback)
            self.mouse_callback_set = True
    
    def draw_selection(self, frame):
        """Dessine la sélection en cours"""
        if self.selecting and self.start_point and self.end_point:
            # Dessiner le rectangle de sélection
            cv2.rectangle(frame, self.start_point, self.end_point, (0, 255, 255), 2)
            
            # Afficher les coordonnées
            text = f"Selection: {abs(self.end_point[0] - self.start_point[0])}x{abs(self.end_point[1] - self.start_point[1])}"
            cv2.putText(frame, text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    def get_roi(self):
        """Retourne la ROI sélectionnée"""
        return self.roi
    
    def reset_roi(self):
        """Remet à zéro la ROI"""
        self.roi = None
        self.selecting = False
        self.start_point = None
        self.end_point = None
        
    def reset(self):
        """Réinitialise la sélection de ROI"""
        self.roi = None
        self.selecting = False
        self.start_point = None
        self.end_point = None


class HeartRateVisualizer:
    """Classe pour la visualisation du rythme cardiaque sur la ROI"""
    
    def __init__(self, pulse_history_size=30):
        self.pulse_history = deque(maxlen=pulse_history_size)
        self.last_pulse_time = 0
        self.pulse_intensity = 0
        
    def update_pulse(self, bpm, green_value=None):
        """Met à jour les données de pulsation"""
        current_time = time.time()
        
        if bpm > 0:
            # Calculer l'intervalle entre les battements
            beat_interval = 60.0 / bpm
            
            # Créer un signal sinusoïdal basé sur le BPM
            pulse_phase = (current_time * 2 * np.pi) / beat_interval
            self.pulse_intensity = int(100 * (1 + np.sin(pulse_phase)))
            
            # Ajouter à l'historique
            self.pulse_history.append({
                'time': current_time,
                'bpm': bpm,
                'intensity': self.pulse_intensity,
                'green_value': green_value
            })
        else:
            self.pulse_intensity = 50  # Valeur neutre
    
    def draw_pulse_overlay(self, frame, roi, current_bpm=0):
        """Dessine une surcouche pulsante sur la ROI - VERSION CORRIGÉE"""
        if roi is None or frame is None:
            return frame
            
        x, y, w, h = roi
        
        # Vérifier les limites strictement
        if (x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0] or
            w <= 0 or h <= 0):
            return frame
        
        try:
            if current_bpm > 0:
                # Couleur basée sur le BPM
                if 60 <= current_bpm <= 90:
                    base_color = (0, 255, 0)  # Vert
                elif current_bpm < 50 or current_bpm > 110:
                    base_color = (0, 0, 255)  # Rouge
                else:
                    base_color = (0, 165, 255)  # Orange
                
                # Intensité pulsante (plus visible)
                alpha = 0.1 + 0.3 * (self.pulse_intensity / 100.0)
                
                # Créer l'overlay sans modifier la frame directement
                overlay = np.zeros_like(frame, dtype=np.uint8)
                overlay[y:y+h, x:x+w] = base_color
                
                # Appliquer l'overlay avec transparence sur toute la frame
                frame = cv2.addWeighted(frame, 1-alpha, overlay, alpha, 0)
                
                # Ajouter un effet de bordure pulsante
                border_thickness = int(2 + 3 * (self.pulse_intensity / 100.0))
                cv2.rectangle(frame, (x, y), (x + w, y + h), base_color, border_thickness)
                
                # Ajouter du texte sur la ROI
                pulse_text = f"♥ {current_bpm}"
                text_x = x + w//2 - 20
                text_y = y + h//2
                cv2.putText(frame, pulse_text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                # ROI inactive - bordure grise
                cv2.rectangle(frame, (x, y), (x + w, y + h), (128, 128, 128), 2)
                cv2.putText(frame, "En attente...", (x + 5, y + h//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        
        except Exception as e:
            # En cas d'erreur, on ne fait rien pour éviter le crash
            pass
            
        return frame
    
    def draw_pulse_graph(self, frame, position=(10, 50), size=(200, 80)):
        """Dessine un petit graphique du rythme cardiaque"""
        if len(self.pulse_history) < 2:
            return
            
        x_start, y_start = position
        width, height = size
        
        # Fond du graphique
        cv2.rectangle(frame, (x_start, y_start), (x_start + width, y_start + height), (30, 30, 30), -1)
        cv2.rectangle(frame, (x_start, y_start), (x_start + width, y_start + height), (100, 100, 100), 1)
        
        # Titre
        cv2.putText(frame, "Rythme Cardiaque", (x_start + 5, y_start - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Dessiner le signal
        if len(self.pulse_history) > 1:
            points = []
            max_intensity = max([p['intensity'] for p in self.pulse_history])
            min_intensity = min([p['intensity'] for p in self.pulse_history])
            
            for i, pulse_data in enumerate(self.pulse_history):
                x_pos = x_start + int(i * width / len(self.pulse_history))
                # Normaliser l'intensité
                if max_intensity > min_intensity:
                    normalized = (pulse_data['intensity'] - min_intensity) / (max_intensity - min_intensity)
                else:
                    normalized = 0.5
                y_pos = y_start + height - int(normalized * (height - 10)) - 5
                points.append((x_pos, y_pos))
            
            # Dessiner les lignes avec dégradé
            for i in range(len(points) - 1):
                # Couleur qui change selon l'intensité
                intensity_ratio = i / len(points)
                color = (
                    int(255 * intensity_ratio),
                    int(255 * (1 - intensity_ratio)),
                    0
                )
                cv2.line(frame, points[i], points[i + 1], color, 2)
        
        # Afficher le BPM actuel
        if self.pulse_history:
            current_data = self.pulse_history[-1]
            bpm_text = f"BPM: {current_data['bpm']}"
            cv2.putText(frame, bpm_text, (x_start + 5, y_start + height - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


class SignalProcessor:
    """Classe pour le traitement du signal PPG"""
    
    def __init__(self, fps, low_freq=0.7, high_freq=4.0, filter_order=4):
        self.fps = fps
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.filter_order = filter_order
        self.nyquist_freq = fps / 2
        
    def apply_bandpass_filter(self, signal):
        """
        Applique un filtre passe-bande au signal
        
        Args:
            signal: Signal d'entrée (array-like)
            
        Returns:
            Signal filtré (numpy array)
        """
        if len(signal) < 60:
            return np.array(signal)
            
        try:
            # Concevoir le filtre passe-bande Butterworth
            sos = scipy.signal.butter(
                self.filter_order,
                [self.low_freq, self.high_freq],
                btype='band',
                fs=self.fps,
                output='sos'
            )
            
            # Appliquer le filtre
            filtered_signal = scipy.signal.sosfilt(sos, signal)
            return filtered_signal
            
        except Exception as e:
            print(f"Erreur lors du filtrage: {e}")
            return np.array(signal)
    
    def calculate_bpm_from_fft(self, signal):
        """
        Calcule le BPM à partir d'un signal en utilisant la FFT
        
        Args:
            signal: Signal PPG (array-like)
            
        Returns:
            BPM estimé (int)
        """
        if len(signal) < 60:
            return 0
            
        # Convertir en array numpy
        signal_array = np.array(signal)
        
        # Supprimer la composante DC
        signal_array = signal_array - np.mean(signal_array)
        
        # Appliquer le filtre passe-bande
        filtered_signal = self.apply_bandpass_filter(signal_array)
        
        # Calculer la FFT
        fft_values = fft(filtered_signal)
        frequencies = fftfreq(len(filtered_signal), 1/self.fps)
        
        # Prendre seulement les fréquences positives
        positive_freq_mask = frequencies > 0
        frequencies = frequencies[positive_freq_mask]
        fft_values = np.abs(fft_values[positive_freq_mask])
        
        # Limiter aux fréquences d'intérêt
        valid_freq_mask = (frequencies >= self.low_freq) & (frequencies <= self.high_freq)
        valid_frequencies = frequencies[valid_freq_mask]
        valid_fft_values = fft_values[valid_freq_mask]
        
        if len(valid_frequencies) == 0:
            return 0
            
        # Trouver le pic dominant
        peak_idx = np.argmax(valid_fft_values)
        peak_frequency = valid_frequencies[peak_idx]
        
        # Convertir en BPM
        bpm = peak_frequency * 60
        
        return int(bpm)
    
    def smooth_signal(self, signal, window_size=5):
        """
        Lisse le signal avec une moyenne mobile
        
        Args:
            signal: Signal d'entrée
            window_size: Taille de la fenêtre de lissage
            
        Returns:
            Signal lissé
        """
        if len(signal) < window_size:
            return signal
            
        return np.convolve(signal, np.ones(window_size)/window_size, mode='valid')


class ROIExtractor:
    """Classe pour l'extraction de la région d'intérêt (ROI)"""
    
    def __init__(self, roi_landmarks, roi_height_factor=0.6, roi_width_factor=0.25):
        self.roi_landmarks = roi_landmarks
        self.roi_height_factor = roi_height_factor
        self.roi_width_factor = roi_width_factor
        
    def extract_roi_from_landmarks(self, landmarks, frame_shape):
        """
        Extrait la ROI du front à partir des landmarks
        
        Args:
            landmarks: Points de repère du visage
            frame_shape: Forme de l'image (height, width)
            
        Returns:
            Coordonnées de la ROI (x, y, w, h) ou None
        """
        h, w = frame_shape[:2]
        
        # Convertir les landmarks normalisés en coordonnées pixel
        roi_points = []
        for landmark_id in self.roi_landmarks:
            if landmark_id < len(landmarks.landmark):
                landmark = landmarks.landmark[landmark_id]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                roi_points.append((x, y))
        
        if not roi_points:
            return None
            
        # Calculer la bounding box
        xs = [p[0] for p in roi_points]
        ys = [p[1] for p in roi_points]
        
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        # Adapter la ROI pour le front
        roi_height = y_max - y_min
        roi_width = x_max - x_min
        
        # Prendre la partie supérieure (front)
        y_max = y_min + int(roi_height * self.roi_height_factor)
        
        # Centrer horizontalement
        center_x = (x_min + x_max) // 2
        half_width = int(roi_width * self.roi_width_factor)
        x_min = center_x - half_width
        x_max = center_x + half_width
        
        # Vérifier les limites
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)
        
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def extract_green_channel_mean(self, frame, roi):
        """
        Extrait la valeur moyenne du canal vert dans la ROI
        
        Args:
            frame: Image d'entrée (BGR)
            roi: Région d'intérêt (x, y, w, h)
            
        Returns:
            Valeur moyenne du canal vert ou None
        """
        if roi is None:
            return None
            
        x, y, w, h = roi
        
        # Vérifier les limites
        if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
            return None
            
        # Extraire la région
        roi_region = frame[y:y+h, x:x+w]
        
        if roi_region.size == 0:
            return None
            
        # Calculer la moyenne du canal vert (index 1 pour BGR)
        green_mean = np.mean(roi_region[:, :, 1])
        
        return green_mean


class ROIStabilizer:
    """Classe pour stabiliser la ROI en cas de mouvement"""
    
    def __init__(self, history_size=5, stability_threshold=20, adjustment_factor=0.3):
        self.history_size = history_size
        self.stability_threshold = stability_threshold
        self.adjustment_factor = adjustment_factor
        self.face_center_history = deque(maxlen=history_size)
        self.stable_roi = None
        
    def stabilize_roi(self, roi, face_center):
        """
        Stabilise la ROI en utilisant l'historique des positions du visage
        
        Args:
            roi: ROI actuelle (x, y, w, h)
            face_center: Centre du visage actuel [x, y]
            
        Returns:
            ROI stabilisée
        """
        if roi is None:
            return None
            
        self.face_center_history.append(face_center)
        
        if len(self.face_center_history) < 3:
            self.stable_roi = roi
            return roi
            
        # Calculer le centre moyen
        avg_center = np.mean(self.face_center_history, axis=0)
        
        # Ajuster la ROI en fonction du mouvement
        if self.stable_roi is not None:
            current_center = np.array(face_center)
            stable_center = np.array([
                self.stable_roi[0] + self.stable_roi[2] // 2,
                self.stable_roi[1] + self.stable_roi[3] // 2
            ])
            
            # Calculer le déplacement
            displacement = avg_center - stable_center
            
            # Vérifier si le mouvement est dans le seuil de stabilité
            if np.linalg.norm(displacement) < self.stability_threshold:
                return self.stable_roi
            
            # Ajuster progressivement
            adjustment = displacement * self.adjustment_factor
            new_roi = (
                int(self.stable_roi[0] + adjustment[0]),
                int(self.stable_roi[1] + adjustment[1]),
                self.stable_roi[2],
                self.stable_roi[3]
            )
            self.stable_roi = new_roi
            return new_roi
        
        self.stable_roi = roi
        return roi
    
    def reset(self):
        """Remet à zéro la stabilisation"""
        self.face_center_history.clear()
        self.stable_roi = None


class Visualizer:
    """Classe pour la visualisation des résultats"""
    
    def __init__(self, colors, display_config):
        self.colors = colors
        self.display_config = display_config
        self.heart_rate_visualizer = HeartRateVisualizer()
        
    def get_bpm_color(self, bpm, thresholds):
        """
        Retourne la couleur correspondant au BPM
        
        Args:
            bpm: Battements par minute
            thresholds: Seuils de BPM
            
        Returns:
            Couleur BGR
        """
        if bpm == 0:
            return self.colors['inactive']
        elif thresholds['normal_min'] <= bpm <= thresholds['normal_max']:
            return self.colors['normal']
        elif bpm < thresholds['critical_min'] or bpm > thresholds['critical_max']:
            return self.colors['critical']
        else:
            return self.colors['warning']
    
    def draw_roi_rectangle(self, frame, roi, color=None, label="ROI", pulse_overlay=False, current_bpm=0, green_value=None):
        """
        Dessine un rectangle autour de la ROI avec option de surcouche pulsante
        
        Args:
            frame: Image où dessiner
            roi: ROI (x, y, w, h)
            color: Couleur du rectangle
            label: Étiquette à afficher
            pulse_overlay: Activer la surcouche pulsante
            current_bpm: BPM actuel pour l'effet de pulsation
            green_value: Valeur du canal vert
        """
        if roi is None:
            return
            
        if color is None:
            color = self.colors['roi']
        
        # Mettre à jour la visualisation du rythme cardiaque
        self.heart_rate_visualizer.update_pulse(current_bpm, green_value)
        
        # Dessiner la surcouche pulsante si activée
        if pulse_overlay:
            self.heart_rate_visualizer.draw_pulse_overlay(frame, roi, current_bpm)
        else:
            # Dessiner le rectangle classique
            x, y, w, h = roi
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Étiquette
            cv2.putText(frame, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def draw_pulse_graph(self, frame, position=(10, 150)):
        """Dessine le graphique du rythme cardiaque"""
        self.heart_rate_visualizer.draw_pulse_graph(frame, position)
    
    def draw_manual_roi_instructions(self, frame):
        """Dessine les instructions pour la sélection manuelle de ROI"""
        instructions = [
            "SELECTION MANUELLE DE ROI:",
            "- Cliquez et glissez pour selectionner",
            "- 'r' pour reset la ROI",
            "- 'a' pour mode automatique",
            "- 'p' pour activer/desactiver surcouche",
            "- 'q' pour quitter"
        ]
        
        start_y = frame.shape[0] - 150
        for i, instruction in enumerate(instructions):
            y_pos = start_y + (i * 20)
            cv2.putText(frame, instruction, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def draw_text_with_background(self, frame, text, position, color=None, bg_color=None):
        """
        Dessine du texte avec un fond pour améliorer la lisibilité
        
        Args:
            frame: Image où dessiner
            text: Texte à afficher
            position: Position (x, y)
            color: Couleur du texte
            bg_color: Couleur du fond
        """
        if color is None:
            color = self.colors['text']
        if bg_color is None:
            bg_color = self.colors['text_bg']
            
        font_scale = self.display_config['font_scale']
        font_thickness = self.display_config['font_thickness']
        
        # Calculer la taille du texte
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        
        # Dessiner le fond
        x, y = position
        cv2.rectangle(frame, (x - 5, y - text_size[1] - 5), 
                     (x + text_size[0] + 5, y + 5), bg_color, -1)
        
        # Dessiner le texte
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, color, font_thickness)
    
    def draw_signal_overlay(self, frame, signal_data, current_bpm=0):
        """
        Dessine le signal rPPG en superposition avec effet de dégradé
        
        Args:
            frame: Image où dessiner
            signal_data: Données du signal à afficher
            current_bpm: BPM actuel pour l'effet de pulsation
        """
        if len(signal_data) < 10:
            return
            
        # Configuration du graphique
        overlay_height = self.display_config['signal_overlay_height']
        overlay_width = self.display_config['signal_overlay_width']
        start_x = frame.shape[1] - overlay_width - 10
        start_y = 10
        
        # Créer une zone semi-transparente
        overlay = np.zeros((overlay_height, overlay_width, 3), dtype=np.uint8)
        
        # Normaliser le signal pour l'affichage
        signal_array = np.array(signal_data)
        if len(signal_array) > 1:
            signal_min, signal_max = np.min(signal_array), np.max(signal_array)
            if signal_max > signal_min:
                signal_normalized = (signal_array - signal_min) / (signal_max - signal_min)
                signal_normalized = signal_normalized * (overlay_height - 20) + 10
            else:
                signal_normalized = np.full_like(signal_array, overlay_height // 2)
            
            # Dessiner le signal avec effet de dégradé
            points = []
            for i, val in enumerate(signal_normalized):
                x = int(i * overlay_width / len(signal_normalized))
                y = int(overlay_height - val)
                points.append((x, y))
            
            # Dessiner les lignes avec effet de pulsation
            for i in range(len(points) - 1):
                if current_bpm > 0:
                    # Couleur qui pulse avec le rythme cardiaque
                    pulse_intensity = int(127 * (1 + np.sin(time.time() * current_bpm / 10)))
                    color = (0, pulse_intensity, 255)
                else:
                    color = (0, 255, 0)
                    
                cv2.line(overlay, points[i], points[i + 1], color, 2)
        
        # Appliquer l'overlay avec transparence
        alpha = self.display_config['transparency']
        roi_overlay = frame[start_y:start_y + overlay_height, start_x:start_x + overlay_width]
        blended = cv2.addWeighted(roi_overlay, alpha, overlay, 1 - alpha, 0)
        frame[start_y:start_y + overlay_height, start_x:start_x + overlay_width] = blended
