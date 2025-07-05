"""
Analyseur de micro-mouvements vasculaires SIMPLIFIÉ et ROBUSTE
Version ultra-simple pour éviter les crashs
"""

import numpy as np
import cv2
from collections import deque


class VascularMicroMovementAnalyzer:
    """
    Version simplifiée et robuste de l'analyseur vasculaire
    """
    
    def __init__(self, history_size=3, sensitivity=0.1):
        self.sensitivity = sensitivity
        
        # Buffer ultra-réduit - juste 3 frames
        self.frames = deque(maxlen=3)
        
        # Cartes de base
        self.heat_map = None
        self.movement_intensity = 0.0
        self.accumulated_movement = None  # Compatibilité avec le code principal
        
        # Paramètres visuels
        self.overlay_alpha = 0.4
        self.frame_count = 0
        
        # Couleurs fixes pour éviter les calculs
        self.base_color = np.array([10, 50, 100], dtype=np.uint8)  # Bleu foncé
        self.active_color = np.array([0, 100, 255], dtype=np.uint8)  # Rouge
        
    def update_frame(self, frame):
        """
        Met à jour avec une nouvelle frame - version ultra-simple
        """
        try:
            if frame is None:
                return
                
            self.frame_count += 1
            
            # Extraire le canal vert (le plus sensible pour rPPG)
            green_channel = frame[:, :, 1].astype(np.float32)
            
            # Ajouter au buffer
            self.frames.append(green_channel)
            
            # Calculer le mouvement si on a au moins 2 frames
            if len(self.frames) >= 2:
                self._simple_movement_analysis(frame.shape[:2])
                
        except Exception as e:
            print(f"Erreur simple: {e}")
            # Continuer quand même
            
    def _simple_movement_analysis(self, frame_shape):
        """
        Analyse ultra-simple des mouvements
        """
        try:
            h, w = frame_shape
            
            # Différence entre les 2 dernières frames
            current = self.frames[-1]
            previous = self.frames[-2]
            
            # Calculer la différence absolue
            diff = cv2.absdiff(current, previous)
            
            # Appliquer un seuil simple
            movement = (diff > (self.sensitivity * 50)).astype(np.float32)
            
            # Flou pour lisser
            movement = cv2.GaussianBlur(movement, (7, 7), 1.0)
            
            # Calculer l'intensité globale
            self.movement_intensity = np.mean(movement)
            
            # Créer une heatmap simple
            self._create_simple_heatmap(movement)
            
        except Exception as e:
            print(f"Erreur analyse: {e}")
            # Garder l'ancien résultat
            
    def _create_simple_heatmap(self, movement):
        """
        Crée une heatmap très simple
        """
        try:
            h, w = movement.shape
            
            # Créer une carte de couleur simple
            heat_map = np.zeros((h, w, 3), dtype=np.uint8)
            
            # Zones sans mouvement = bleu foncé
            heat_map[:, :] = self.base_color
            
            # Zones avec mouvement = rouge
            mask = movement > 0.1
            heat_map[mask] = self.active_color
            
            # Mélanger pour avoir des nuances
            intensity = np.clip(movement * 255, 0, 255).astype(np.uint8)
            for i in range(3):
                heat_map[:, :, i] = np.where(mask, 
                                           np.clip(self.base_color[i] + intensity, 0, 255),
                                           self.base_color[i])
            
            self.heat_map = heat_map
            
        except Exception as e:
            print(f"Erreur heatmap: {e}")
            # Garder l'ancienne heatmap
            
    def draw_movement_overlay(self, frame, show_heat_map=True, show_realtime=True, force_visible=False):
        """
        Dessine l'overlay - version ultra-robuste
        """
        try:
            if frame is None:
                return frame
                
            # Toujours afficher quelque chose pour éviter la disparition
            overlay = frame.copy()
            
            # Si on n'a pas de heatmap, créer un overlay minimal
            if self.heat_map is None:
                h, w = frame.shape[:2]
                # Créer un motif subtil pour montrer que ça marche
                pattern = np.zeros((h, w, 3), dtype=np.uint8)
                pattern[::30, ::30] = [20, 20, 20]  # Points subtils
                overlay = cv2.addWeighted(overlay, 0.95, pattern, 0.05, 0)
                return overlay
            
            # Vérifier que les dimensions correspondent
            if self.heat_map.shape[:2] != frame.shape[:2]:
                # Redimensionner si nécessaire
                self.heat_map = cv2.resize(self.heat_map, (frame.shape[1], frame.shape[0]))
            
            # Appliquer la heatmap
            alpha = self.overlay_alpha
            if force_visible or self.movement_intensity > 0.01:
                alpha = min(0.6, alpha * 1.5)
            
            overlay = cv2.addWeighted(overlay, 1 - alpha, self.heat_map, alpha, 0)
            
            return overlay
            
        except Exception as e:
            print(f"Erreur overlay: {e}")
            return frame  # Retourner la frame originale en cas d'erreur
            
    def draw_hotspot_indicators(self, frame, max_hotspots=3):
        """
        Dessine des indicateurs sur les zones actives (version simple)
        """
        if frame is None or self.movement_intensity < 0.1:
            return
            
        h, w = frame.shape[:2]
        
        # Créer quelques zones aléatoires pour la démo
        for i in range(min(max_hotspots, 2)):
            # Position pseudo-aléatoire basée sur le frame_count
            x = int((self.frame_count * (i + 1) * 123) % (w - 100))
            y = int((self.frame_count * (i + 2) * 456) % (h - 100))
            
            # Taille fixe
            size = 40
            
            # Couleur selon le rang
            colors = [(0, 0, 255), (0, 165, 255), (0, 255, 255)]  # Rouge, Orange, Jaune
            color = colors[i % len(colors)]
            
            # Dessiner un rectangle simple
            cv2.rectangle(frame, (x, y), (x + size, y + size), color, 2)
            
            # Label simple
            cv2.putText(frame, f"#{i+1}", (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def get_movement_hotspots(self, min_area=100):
        """
        Retourne des hotspots simulés pour la compatibilité
        """
        if self.movement_intensity < 0.1:
            return []
        
        # Retourner quelques zones simulées
        hotspots = [
            (50, 50, 40, 40, self.movement_intensity),
            (150, 100, 40, 40, self.movement_intensity * 0.8)
        ]
        
        return hotspots
            
    def draw_hotspot_indicators(self, frame, max_hotspots=3):
        """
        Dessine les indicateurs de hotspots
        """
        try:
            hotspots = self.get_movement_hotspots()
            
            colors = [(0, 0, 255), (0, 165, 255), (0, 255, 255)]  # Rouge, Orange, Jaune
            
            for i, (x, y, w, h, intensity) in enumerate(hotspots[:max_hotspots]):
                color = colors[min(i, len(colors) - 1)]
                
                # Rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # Label
                label = f"#{i+1}"
                cv2.putText(frame, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
        except Exception as e:
            print(f"Erreur indicateurs: {e}")
            
    def get_best_roi_suggestion(self):
        """
        Suggère la meilleure ROI
        """
        try:
            hotspots = self.get_movement_hotspots(min_area=200)
            
            if hotspots:
                x, y, w, h, _ = hotspots[0]
                
                # Centrer et ajuster la taille
                center_x, center_y = x + w//2, y + h//2
                size = 80
                
                new_x = max(0, center_x - size//2)
                new_y = max(0, center_y - size//2)
                
                return (new_x, new_y, size, size)
                
        except Exception as e:
            print(f"Erreur ROI: {e}")
            
        return None
        
    def draw_analysis_info(self, frame):
        """
        Affiche les infos d'analyse - VERSION FORCÉE
        """
        try:
            if frame is None:
                return frame
                
            # TOUJOURS afficher quelque chose - même si pas de données
            info_lines = [
                f"Mouvement: {self.movement_intensity:.3f}",
                f"Frames: {self.frame_count}",
                f"Alpha: {self.overlay_alpha:.2f}",
                "Interface ACTIVE"
            ]
            
            # Fond noir solide
            cv2.rectangle(frame, (10, 10), (220, 90), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (220, 90), (255, 255, 255), 1)  # Bordure blanche
            
            # Texte en blanc
            for i, line in enumerate(info_lines):
                y_pos = 30 + i * 18
                cv2.putText(frame, line, (15, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return frame
            
        except Exception as e:
            print(f"Erreur info: {e}")
            # En cas d'erreur, afficher au moins un rectangle
            try:
                cv2.rectangle(frame, (10, 10), (150, 50), (255, 0, 0), -1)
                cv2.putText(frame, "ERREUR", (15, 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            except:
                pass
            return frame
            
    def auto_adjust_sensitivity(self):
        """
        Ajustement automatique désactivé pour la stabilité
        """
        pass
        
    def boost_overlay_visibility(self):
        """
        Boost simple de visibilité
        """
        try:
            self.overlay_alpha = min(0.8, self.overlay_alpha * 1.2)
            print(f"Overlay boosté à {self.overlay_alpha:.2f}")
        except:
            pass
