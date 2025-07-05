"""
Test simple pour vérifier les fonctionnalités de fenêtre redimensionnable
"""
import cv2
import numpy as np
import time

def test_resizable_window():
    """Test de la fenêtre redimensionnable"""
    # Créer une caméra factice
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Caméra non disponible - Utilisation d'une image factice")
        cap = None
    
    # Créer la fenêtre avec des propriétés redimensionnables
    window_name = '💗 Test Fenêtre Redimensionnable'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)  # Taille par défaut
    
    is_fullscreen = False
    
    print("🚀 === TEST FENÊTRE REDIMENSIONNABLE ===")
    print("🎯 CONTRÔLES :")
    print("🖥️ 'F' : Basculer plein écran")
    print("🚪 'ESC' ou 'Q' : Quitter")
    print("💡 ASTUCE : Tirez les bords pour redimensionner !")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Obtenir une frame
            if cap:
                ret, frame = cap.read()
                if not ret:
                    # Créer une frame factice
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                # Créer une frame factice animée
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                
            # Ajouter du contenu à la frame
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Dessiner des informations
            cv2.putText(frame, f"Frame: {frame_count}", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Temps: {elapsed:.1f}s", (20, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            
            # Mode plein écran
            if is_fullscreen:
                cv2.putText(frame, "MODE PLEIN ECRAN", (20, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "MODE FENETRE", (20, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            # Instructions
            cv2.putText(frame, "F = Plein ecran | ESC/Q = Quitter", (20, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            # Dessiner un motif animé
            center_x, center_y = 320, 240
            radius = int(50 + 30 * np.sin(frame_count * 0.1))
            cv2.circle(frame, (center_x, center_y), radius, (0, 255, 0), 3)
            
            # Afficher
            cv2.imshow(window_name, frame)
            
            # Gestion des événements clavier
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # ESC
                break
            elif key == ord('f') or key == ord('F'):  # Plein écran
                if is_fullscreen:
                    # Revenir au mode fenêtré
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, 800, 600)
                    is_fullscreen = False
                    print("🖥️ Mode fenêtré activé")
                else:
                    # Passer en plein écran
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    is_fullscreen = True
                    print("🖥️ Mode plein écran activé")
        
    except KeyboardInterrupt:
        print("\n🛑 Interruption par l'utilisateur")
    
    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n📊 === STATISTIQUES ===")
        print(f"⏱️ Durée: {elapsed:.1f}s")
        print(f"🎬 Frames: {frame_count}")
        print(f"📺 FPS moyen: {fps:.1f}")
        print("✅ Test terminé avec succès!")

if __name__ == "__main__":
    test_resizable_window()
