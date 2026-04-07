import cv2
import numpy as np
import json
import os
import glob
import argparse
from ultralytics import YOLO
from tqdm import tqdm

# Diccionario con las 29 coordenadas geométricas base (105x68 métrico)
# Origen (0,0) en la esquina superior izquierda. X = 0 a 105, Y = 0 a 68.
FIFA_29_PTS = {
    0: (0.0, 0.0),                               # sideline_top_left
    1: (0.0, 13.85),                             # big_rect_left_top_pt1
    2: (16.5, 13.85),                            # big_rect_left_top_pt2
    3: (0.0, 54.15),                             # big_rect_left_bottom_pt1
    4: (16.5, 54.15),                            # big_rect_left_bottom_pt2
    5: (0.0, 24.85),                             # small_rect_left_top_pt1
    6: (5.5, 24.85),                             # small_rect_left_top_pt2
    7: (0.0, 43.15),                             # small_rect_left_bottom_pt1
    8: (5.5, 43.15),                             # small_rect_left_bottom_pt2
    9: (0.0, 68.0),                              # sideline_bottom_left
    10: (20.15, 34.0),                           # left_semicircle_right (11m + 9.15m radio)
    11: (52.5, 0.0),                             # center_line_top
    12: (52.5, 68.0),                            # center_line_bottom
    13: (52.5, 24.85),                           # center_circle_top
    14: (52.5, 43.15),                           # center_circle_bottom
    15: (52.5, 34.0),                            # field_center
    16: (105.0, 0.0),                            # sideline_top_right
    17: (105.0, 13.85),                          # big_rect_right_top_pt1
    18: (88.5, 13.85),                           # big_rect_right_top_pt2
    19: (105.0, 54.15),                          # big_rect_right_bottom_pt1
    20: (88.5, 54.15),                           # big_rect_right_bottom_pt2
    21: (105.0, 24.85),                          # small_rect_right_top_pt1
    22: (99.5, 24.85),                           # small_rect_right_top_pt2
    23: (105.0, 43.15),                          # small_rect_right_bottom_pt1
    24: (99.5, 43.15),                           # small_rect_right_bottom_pt2
    25: (105.0, 68.0),                           # sideline_bottom_right
    26: (84.85, 34.0),                           # right_semicircle_left (105 - 11 - 9.15)
    27: (43.35, 34.0),                           # center_circle_left (52.5 - 9.15)
    28: (61.65, 34.0),                           # center_circle_right (52.5 + 9.15)
}

def draw_pitch_cv2(scale=10, margin=50):
    """
    Dibuja un minimapa 2D usando OpenCV para máxima velocidad.
    Retorna la imagen BGR base.
    """
    w = int(105 * scale)
    h = int(68 * scale)
    img_w, img_h = w + 2*margin, h + 2*margin
    
    # Césped verde
    pitch = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    pitch[:] = (60, 120, 50) # BGR
    
    white = (255, 255, 255)
    thictness = 2
    
    # Función lambda para escalar
    pt = lambda x, y: (int(x * scale) + margin, int(y * scale) + margin)
    
    # Bordes exteriores
    cv2.rectangle(pitch, pt(0,0), pt(105, 68), white, thictness)
    
    # Línea central
    cv2.line(pitch, pt(52.5, 0), pt(52.5, 68), white, thictness)
    
    # Círculo central
    cv2.circle(pitch, pt(52.5, 34), int(9.15 * scale), white, thictness)
    # Punto central
    cv2.circle(pitch, pt(52.5, 34), 2, white, -1)
    
    # Áreas Pequeñas
    cv2.rectangle(pitch, pt(0, 24.85), pt(5.5, 43.15), white, thictness)
    cv2.rectangle(pitch, pt(105-5.5, 24.85), pt(105, 43.15), white, thictness)
    
    # Áreas Grandes
    cv2.rectangle(pitch, pt(0, 13.85), pt(16.5, 54.15), white, thictness)
    cv2.rectangle(pitch, pt(105-16.5, 13.85), pt(105, 54.15), white, thictness)
    
    # Puntos de penal
    cv2.circle(pitch, pt(11, 34), 2, white, -1)
    cv2.circle(pitch, pt(105-11, 34), 2, white, -1)
    
    # Semicírculos (aproximado usando arcos)
    cv2.ellipse(pitch, pt(11, 34), (int(9.15*scale), int(9.15*scale)), 0, -53.13, 53.13, white, thictness)
    cv2.ellipse(pitch, pt(105-11, 34), (int(9.15*scale), int(9.15*scale)), 0, 126.87, 233.13, white, thictness)

    return pitch

def project_point(x, y, H, scale, margin):
    """Aplica la matriz de homografía H y escapa a la vista del minimapa"""
    pts = np.array([[[x, y]]], dtype="float32")
    proj = cv2.perspectiveTransform(pts, H)
    px, py = proj[0][0]
    return int(px * scale) + margin, int(py * scale) + margin

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_dir", type=str, required=True)
    parser.add_argument("--detections", type=str, required=True)
    parser.add_argument("--trajectory", type=str, required=True)
    parser.add_argument("--keypoints_model", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    print(f"Cargando YOLOv11-Pose Soccana desde {args.keypoints_model}...")
    model_kp = YOLO(args.keypoints_model)

    print(f"Cargando datos de tracking...")
    with open(args.detections, "r") as f:
        detections = {d["frame_id"]: d for d in json.load(f)}
    with open(args.trajectory, "r") as f:
        trajectory = {t["frame_id"]: t for t in json.load(f)}

    img_dir = os.path.join(args.sequence_dir, "img1")
    images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))

    if not images:
        print("Error: No hay imágenes en la ruta de la secuencia.")
        return

    # Video Setup
    frame_0 = cv2.imread(images[0])
    h_ori, w_ori = frame_0.shape[:2]

    # Pre-render pitch (escala 1 metro = 10 pixels p. ej)
    scale = 8
    margin = 40
    base_pitch = draw_pitch_cv2(scale, margin)
    h_pitch, w_pitch = base_pitch.shape[:2]

    # Redimensionar el original para match con el minimapa (side-by-side)
    target_video_h = max(h_ori, h_pitch)
    scale_ori = target_video_h / h_ori
    new_w_ori = int(w_ori * scale_ori)
    
    out_w = new_w_ori + w_pitch
    out_h = target_video_h

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(args.output, fourcc, 25.0, (out_w, out_h))

    last_valid_H = None

    print(f"Iniciando Renderización Side-By-Side (Total frames: {len(images)})")
    
    for img_path in tqdm(images):
        frame_id = int(os.path.splitext(os.path.basename(img_path))[0])
        frame = cv2.imread(img_path)
        
        # 1. Extraer 29 puntos clave de la matriz del campo
        results_kp = model_kp.predict(frame, verbose=False, device='cuda')
        kpts = results_kp[0].keypoints.data[0].cpu().numpy() # [29, 3]

        src_pts = []
        dst_pts = []

        for i in range(29):
            x, y, conf = kpts[i]
            if conf > 0.5: # Umbral de confianza
                src_pts.append([x, y])
                dst_pts.append(FIFA_29_PTS[i])
        
        # 2. Computar o Reciclar Homografía
        H = None
        if len(src_pts) >= 4:
            src_arr = np.array(src_pts, dtype=np.float32)
            dst_arr = np.array(dst_pts, dtype=np.float32)
            H, _ = cv2.findHomography(src_arr, dst_arr, cv2.RANSAC, 5.0)
        
        if H is not None:
            last_valid_H = H
        else:
            H = last_valid_H

        # Preparar marcos visuales
        pitch_frame = base_pitch.copy()
        
        # Dibujar Kpts en el frame original por validación
        for x, y in src_pts:
            cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 255), -1)

        # 3. Proyectar y Renderizar Jugadores
        frame_data = detections.get(frame_id, {"players": []})
        for p in frame_data["players"]:
            x_min, y_min, x_max, y_max = p["x_min"], p["y_min"], p["x_max"], p["y_max"]
            track_id = p["track_id"]
            
            # Dibujar rect original
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)
            
            # Anclaje Inferior (Z=0 aprox)
            x_center = (x_min + x_max) / 2.0
            y_bottom = y_max
            
            if H is not None:
                px, py = project_point(x_center, y_bottom, H, scale, margin)
                # Bounds check
                if 0 <= px < w_pitch and 0 <= py < h_pitch:
                    cv2.circle(pitch_frame, (px, py), 6, (255, 0, 0), -1)
                    cv2.putText(pitch_frame, str(track_id), (px+8, py), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 4. Proyectar y Renderizar Balón (Viterbi)
        ball_data = trajectory.get(frame_id)
        if ball_data and H is not None:
            bx, by = ball_data["x"], ball_data["y"]
            is_dummy = ball_data.get("is_dummy", False)
            color_ball = (0, 0, 255) if is_dummy else (0, 255, 255)
            
            # Dibujar en video
            cv2.circle(frame, (int(bx), int(by)), 5, color_ball, -1)
            
            # En minimapa, advertencia visual temporal: el balón vuela, homografía puede mentir (fallo Z!=0)
            px, py = project_point(bx, by, H, scale, margin)
            if 0 <= px < w_pitch and 0 <= py < h_pitch:
                cv2.circle(pitch_frame, (px, py), 5, (0, 165, 255), -1) # Naranja
        
        # 5. Concatenar y Escribir
        frame_resized = cv2.resize(frame, (new_w_ori, target_video_h))
        pitch_resized = cv2.resize(pitch_frame, (w_pitch, target_video_h)) # No debería redimensionarse mucho, h_pitch ya es 68*8
        
        # Superponer pitch frame centrado o escalado
        pad_top = (target_video_h - h_pitch) // 2
        pad_bot = target_video_h - h_pitch - pad_top
        pitch_padded = cv2.copyMakeBorder(pitch_frame, pad_top, pad_bot, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])

        final_frame = np.hstack((frame_resized, pitch_padded))
        out_video.write(final_frame)

    out_video.release()
    print(f"Proceso finalizado. Video guardado en: {args.output}")

if __name__ == '__main__':
    main()
