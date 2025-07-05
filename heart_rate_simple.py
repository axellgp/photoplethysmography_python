"""
Détecteur de rythme cardiaque SIMPLE - VERSION OPENCV
=====================================================

Version compatible Python 3.13 utilisant uniquement OpenCV
✅ Fenêtre redimensionnable et mode plein écran
✅ Détection de visage avec OpenCV Haar Cascade
✅ Calcul BPM basique mais fonctionnel
✅ Interface scalable
"""

import cv2
import numpy as np
from collections import deque
import time

class SimpleHeartRateDetector:
    """Détecteur de rythme             if self.bpm_history:
                print(f"💓 BPM moyen: {np.mean(list(self.bpm_history)):.1f}")
                print(f"💗 BPM final: {self.current_bpm:.1f}")diaque simple avec OpenCV"""
    
    def __init__(self):
        # Détecteur de visage
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Variables d'état
        self.mode = "auto"  # "auto" ou "manual"
        self.current_roi = None
        self.is_selecting_roi = False
        self.is_fullscreen = False
        
        # Signal processing
        self.signal_buffer = deque(maxlen=150)  # 5 secondes à 30fps
        self.bpm_buffer = deque(maxlen=10)
        
        # Métriques
        self.current_bpm = 0
        self.bpm_history = deque(maxlen=100)
        self.signal_quality_history = deque(maxlen=30)
        
        # Compteurs
        self.frame_count = 0
        self.face_detection_count = 0
        self.start_time = time.time()
        
        # Options d'affichage
        self.ui_styles = ["minimal", "complete", "fullscreen"]
        self.ui_style = "complete"
        
        # Notifications
        self.notification_queue = deque(maxlen=5)
        
        # Couleurs
        self.colors = {
            'good': (0, 255, 0),
            'warning': (0, 255, 255),
            'danger': (0, 0, 255),
            'inactive': (128, 128, 128),
            'text': (255, 255, 255),
            'accent': (255, 100, 100)
        }
        
        # ROI selection
        self.selecting = False
        self.start_point = None
        self.end_point = None
        
    def add_notification(self, message, msg_type="info"):
        """Ajoute une notification"""
        self.notification_queue.append({
            'message': message,
            'type': msg_type,
            'time': time.time()
        })
        
    def mouse_callback(self, event, x, y, flags, param):
        """Callback souris pour sélection ROI"""
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
                    self.current_roi = (roi_x, roi_y, roi_w, roi_h)
                    self.add_notification("ROI sélectionnée", "success")
    
    def detect_face_roi(self, frame):
        """Détecte le visage et retourne ROI du front"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Prendre le plus grand visage
            x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
            
            # ROI du front (tiers supérieur)
            forehead_y = y + int(h * 0.1)
            forehead_h = int(h * 0.3)
            forehead_x = x + int(w * 0.25)
            forehead_w = int(w * 0.5)
            
            # Limiter la taille
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
    
    def calculate_bpm(self):
        """Calcule le BPM à partir du signal"""
        if len(self.signal_buffer) < 60:  # Minimum 2 secondes
            return 0
        
        # Convertir en array
        signal = np.array(list(self.signal_buffer))
        
        # Détrendre le signal
        signal = signal - np.mean(signal)
        
        # FFT
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/30)  # 30 fps
        
        # Filtrer les fréquences cardiaques (0.8-3 Hz = 48-180 BPM)
        valid_freqs = (freqs >= 0.8) & (freqs <= 3.0)
        
        if not np.any(valid_freqs):
            return 0
        
        # Trouver le pic dominant
        power = np.abs(fft[valid_freqs])
        if len(power) == 0:
            return 0
            
        peak_idx = np.argmax(power)
        peak_freq = freqs[valid_freqs][peak_idx]
        
        bpm = peak_freq * 60
        
        # Validation
        if 40 <= bpm <= 200:
            self.bpm_buffer.append(bpm)
            if len(self.bpm_buffer) >= 3:
                return np.median(list(self.bpm_buffer))
            return bpm
        
        return 0
    
    def get_bpm_color(self, bpm):
        """Retourne la couleur selon le BPM"""
        if bpm == 0:
            return self.colors['inactive']
        elif bpm < 50 or bpm > 120:
            return self.colors['danger']
        elif 60 <= bpm <= 90:
            return self.colors['good']
        else:
            return self.colors['warning']
    
    def calculate_signal_quality(self, roi_frame):
        """Calcule la qualité du signal"""
        if roi_frame.size == 0:
            return 0
        
        green_channel = roi_frame[:, :, 1]
        mean_val = np.mean(green_channel)
        var_val = np.var(green_channel)
        
        quality = min(100, max(0, (var_val / mean_val) * 100 if mean_val > 0 else 0))
        return quality
    
    def process_frame(self, frame):
        """Traite une frame"""
        if frame is None:
            return None
        
        self.frame_count += 1
        
        # Gestion ROI
        if self.mode == "auto":
            detected_roi = self.detect_face_roi(frame)
            if detected_roi:
                self.current_roi = detected_roi
                self.face_detection_count += 1
        
        # Traitement signal
        signal_quality = 0
        if self.current_roi:
            x, y, w, h = self.current_roi
            roi_frame = frame[y:y+h, x:x+w]
            
            if roi_frame.size > 0:
                # Qualité du signal
                signal_quality = self.calculate_signal_quality(roi_frame)
                self.signal_quality_history.append(signal_quality)
                
                # Signal rPPG (canal vert)
                signal = np.mean(roi_frame[:, :, 1])
                self.signal_buffer.append(signal)
                
                # Calculer BPM
                new_bpm = self.calculate_bpm()
                if new_bpm > 0:
                    self.current_bpm = new_bpm
                    self.bpm_history.append(new_bpm)
                
                # Dessiner ROI
                self.draw_roi(frame, (x, y, w, h), signal_quality)
        
        # Interface utilisateur
        frame = self.draw_ui(frame, signal_quality)
        
        # Notifications
        frame = self.draw_notifications(frame)
        
        return frame
    
    def draw_roi(self, frame, roi, signal_quality):
        """Dessine la ROI"""
        x, y, w, h = roi
        
        # Couleur selon qualité
        if signal_quality > 70:
            color = self.colors['good']
        elif signal_quality > 40:
            color = self.colors['warning']
        else:
            color = self.colors['danger']
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"Q: {signal_quality:.0f}%", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def draw_ui(self, frame, signal_quality):
        """Dessine l'interface"""
        if self.ui_style == "minimal":
            return self.draw_minimal_ui(frame)
        elif self.ui_style == "fullscreen":
            return self.draw_fullscreen_ui(frame, signal_quality)
        else:
            return self.draw_complete_ui(frame, signal_quality)
    
    def draw_minimal_ui(self, frame):
        """Interface minimale"""
        bpm_color = self.get_bpm_color(self.current_bpm)
        bmp_text = f"BPM: {self.current_bpm:.1f}" if self.current_bpm > 0 else "BPM: --"
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (200, 50), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
        
        cv2.putText(frame, bmp_text, (15, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, bpm_color, 2)
        
        return frame
    
    def draw_complete_ui(self, frame, signal_quality):
        """Interface complète"""
        h, w = frame.shape[:2]
        
        # Panel gauche
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (320, 140), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        cv2.rectangle(frame, (10, 10), (320, 140), (100, 100, 100), 2)
        
        # Titre
        cv2.putText(frame, "HEART RATE MONITOR", (15, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['accent'], 2)
        
        # BPM
        bpm_color = self.get_bpm_color(self.current_bpm)
        bmp_text = f"BPM: {self.current_bpm:.1f}" if self.current_bpm > 0 else "BPM: Calcul..."
        cv2.putText(frame, bmp_text, (15, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, bpm_color, 3)
        
        # Stats
        if self.bmp_history:
            avg_bpm = np.mean(list(self.bmp_history))
            cv2.putText(frame, f"Moyenne: {avg_bmp:.1f}", (15, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        if self.signal_quality_history:
            avg_quality = np.mean(list(self.signal_quality_history))
            cv2.putText(frame, f"Qualite: {avg_quality:.0f}%", (15, 105), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        cv2.putText(frame, f"Mode: {self.mode.upper()}", (15, 125), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['accent'], 1)
        
        # Panel contrôles
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - 220, 10), (w - 10, 200), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        cv2.rectangle(frame, (w - 220, 10), (w - 10, 200), (100, 100, 100), 2)
        
        cv2.putText(frame, "CONTROLES", (w - 215, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['accent'], 1)
        
        controls = [
            "A: Mode auto",
            "M: Mode manuel", 
            "F: Style UI",
            "F11: Plein ecran",
            "R: Reset",
            "Q: Quitter"
        ]
        
        for i, control in enumerate(controls):
            cv2.putText(frame, control, (w - 210, 50 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.colors['text'], 1)
        
        return frame
    
    def draw_fullscreen_ui(self, frame, signal_quality):
        """Interface plein écran"""
        h, w = frame.shape[:2]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
        
        # BPM géant
        bpm_color = self.get_bpm_color(self.current_bpm)
        bmp_text = f"{self.current_bpm:.1f}" if self.current_bpm > 0 else "--"
        
        font_scale = min(w / 400, h / 300)
        cv2.putText(frame, bmp_text, (w // 2 - 80, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale * 2, bpm_color, 4)
        
        cv2.putText(frame, "BPM", (w // 2 - 30, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['text'], 2)
        
        return frame
    
    def draw_notifications(self, frame):
        """Dessine les notifications"""
        current_time = time.time()
        
        # Nettoyer les anciennes
        self.notification_queue = deque([
            n for n in self.notification_queue 
            if current_time - n['time'] < 3.0
        ], maxlen=5)
        
        # Dessiner
        for i, notification in enumerate(self.notification_queue):
            message = notification['message']
            age = current_time - notification['time']
            alpha = max(0.3, 1.0 - age / 3.0)
            
            color = tuple(int(c * alpha) for c in self.colors['text'])
            cv2.putText(frame, message, (10, 180 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return frame
    
    def toggle_fullscreen(self, window_name):
        """Bascule plein écran"""
        if self.is_fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 800, 600)
            self.is_fullscreen = False
            self.add_notification("Mode fenêtré activé", "info")
        else:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            self.is_fullscreen = True
            self.add_notification("Mode plein écran activé", "info")
    
    def run(self):
        """Lance l'application"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Impossible d'accéder à la caméra")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        window_name = '💗 Heart Rate Monitor - Simple'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
        
        print("🚀 === DÉTECTEUR DE RYTHME CARDIAQUE SIMPLE ===")
        print("✨ Version compatible Python 3.13")
        print()
        print("🎯 CONTRÔLES :")
        print("🔄 'A' : Mode automatique")
        print("✋ 'M' : Mode manuel")
        print("🎨 'F' : Style UI")
        print("🖥️ 'F11' ou 'Shift+F' : Plein écran")
        print("🔄 'R' : Reset")
        print("🚪 'ESC' ou 'Q' : Quitter")
        print()
        print("💡 ASTUCE : Fenêtre redimensionnable !")
        print()
        
        self.add_notification("Détecteur simple lancé!", "success")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed_frame = self.process_frame(frame)
                
                if processed_frame is not None:
                    cv2.imshow(window_name, processed_frame)
                else:
                    cv2.imshow(window_name, frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # ESC
                    break
                elif key == ord('a'):
                    self.mode = "auto"
                    self.is_selecting_roi = False
                    self.add_notification("Mode automatique", "info")
                elif key == ord('m'):
                    self.mode = "manual"
                    self.is_selecting_roi = True
                    self.add_notification("Mode manuel", "info")
                elif key == ord('f'):
                    current_idx = self.ui_styles.index(self.ui_style)
                    self.ui_style = self.ui_styles[(current_idx + 1) % len(self.ui_styles)]
                    self.add_notification(f"Style UI: {self.ui_style}", "info")
                elif key == 225 or key == ord('F'):  # F11
                    self.toggle_fullscreen(window_name)
                elif key == ord('r'):
                    self.signal_buffer.clear()
                    self.bpm_buffer.clear()
                    self.bpm_history.clear()
                    self.current_bpm = 0
                    self.add_notification("Reset!", "success")
                
                # Gestion souris pour mode manuel
                if self.mode == "manual" and self.is_selecting_roi:
                    cv2.setMouseCallback(window_name, self.mouse_callback)
        
        except KeyboardInterrupt:
            print("\n🛑 Interruption par l'utilisateur")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            elapsed_time = time.time() - self.start_time
            fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            
            print(f"\n📊 === STATISTIQUES FINALES ===")
            print(f"⏱️ Durée: {elapsed_time:.1f}s")
            print(f"🎬 Frames: {self.frame_count}")
            print(f"📺 FPS moyen: {fps:.1f}")
            print(f"👤 Faces détectées: {self.face_detection_count}")
            if self.bpm_history:
                print(f"💓 BPM moyen: {np.mean(list(self.bmp_history)):.1f}")
                print(f"💗 BPM final: {self.current_bpm:.1f}")
            print("✅ Session terminée avec succès!")


def main():
    """Point d'entrée"""
    try:
        detector = SimpleHeartRateDetector()
        detector.run()
    except Exception as e:
        print(f"❌ Erreur fatale: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
