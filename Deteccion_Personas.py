import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from shapely.geometry import Polygon, Point
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from collections import defaultdict
from pathlib import Path  


# Diccionario para mantener el historial de seguimiento de cada objeto
track_history = defaultdict(list)

# Definición de regiones de conteo con polígonos, nombres, contadores, etc.
counting_regions = [
    {
        "name": "ZONA 1",
        "polygon": Polygon([(10, 240), (10, 170), (450, 200), (440, 350), (200, 260)]),  # Puntos del polígono
        "counts": 0,
        "dragging": False,
        "region_color": (255, 42, 4),  # Valor BGR
        "text_color": (255, 255, 255),  # Color del texto de la región
    },
    {
        "name": "ZONA 2",
        "polygon": Polygon([(440, 350), (700, 250), (755, 280), (755, 480)]),  # Puntos del polígono
        "counts": 0,
        "dragging": False,
        "region_color": (37, 255, 225),  # Valor BGR
        "text_color": (0, 0, 0),  # Color del texto de la región
    },
    {
        "name": "ZONA 3",
        "polygon": Polygon([(755, 150), (755, 70), (450, 200), (700, 250), (680, 210)]),  # Puntos del polígono
        "counts": 0,
        "dragging": False,
        "region_color": (100, 200, 50),  # Valor BGR
        "text_color": (255, 255, 255),  # Color del texto de la región   
    },
    {
        "name": "ZONA 4",
        "polygon": Polygon([(455, 205), (685, 250), (445, 345)]),  # Puntos del polígono
        "counts": 0,
        "dragging": False,
        "region_color": (80, 130, 180),  # Valor BGR
        "text_color": (255, 255, 255),  # Color del texto de la región
    }
]

# Función principal para ejecutar la detección de objetos en el video
def run_video():
    weights = "yolov8n.pt"
    source = "video.mp4"
    device = "cpu"
    view_img = True
    classes = [0]

    vid_frame_count = 0

    # Comprueba la existencia de la ruta de origen
    if not Path(source).exists():
        raise FileNotFoundError(f"No existe '{source}'")

    # Configura el modelo YOLO
    model = YOLO(f"{weights}")
    model.to("cuda") if device == "0" else model.to("cpu")

    # Extrae los nombres de las clases
    names = model.model.names

    # Configuración del video
    videocapture = cv2.VideoCapture(source)
    frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
    fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*"mp4v")

    # Itera sobre los fotogramas del video
    while videocapture.isOpened():
        success, frame = videocapture.read()
        if not success:
            break
        vid_frame_count += 1

        # Extrae los resultados de la detección de objetos
        results = model.track(frame, persist=True, classes=classes)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()

            person_counter = 1

            annotator = Annotator(frame, line_width=2, example=str(names))

            for box, track_id, cls in zip(boxes, track_ids, clss):

                if cls == 0:  # Suponiendo que la clase '0' corresponde a personas
                    if track_id not in track_history:
                        track_history[track_id] = []
                    if track_id not in track_history:
                        track_history[track_id].append(person_counter)
                        person_counter += 1                
                annotator.box_label(box, f"{str(names[cls])}a ID: {track_id}", color=colors(cls, True))
                bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2  # Centro del cuadro delimitador

                track = track_history[track_id]  # Trama del historial de seguimiento
                track.append((float(bbox_center[0]), float(bbox_center[1])))
                if len(track) > 30:
                    track.pop(0)
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=colors(cls, True), thickness=2)

                # Comprueba si la detección está dentro de alguna región
                for region in counting_regions:
                    if region["polygon"].contains(Point((bbox_center[0], bbox_center[1]))):
                        region["counts"] += 1

        # Dibuja las regiones (Polígonos/Rectángulos)
        for region in counting_regions:
            region_label = str(region["counts"])
            region_color = region["region_color"]
            region_text_color = region["text_color"]

            polygon_coords = np.array(region["polygon"].exterior.coords, dtype=np.int32)
            centroid_x, centroid_y = int(region["polygon"].centroid.x), int(region["polygon"].centroid.y)

            text_size, _ = cv2.getTextSize(
                region_label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=2
            )
            text_x = centroid_x - text_size[0] // 2
            text_y = centroid_y + text_size[1] // 2
            cv2.rectangle(
                frame,
                (text_x - 5, text_y - text_size[1] - 5),
                (text_x + text_size[0] + 5, text_y + 5),
                region_color,
                -1,
            )
            cv2.putText(
                frame, region_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_text_color, 2
            )
            cv2.polylines(frame, [polygon_coords], isClosed=True, color=region_color, thickness=2)

        if view_img:
            cv2.imshow("Procesamiento del Video", frame)

        for region in counting_regions:  # Reinicializa el contador para cada región
            region["counts"] = 0

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    del vid_frame_count
    videocapture.release()
    cv2.destroyAllWindows()

def start_detection():
    run_video()

def create_ui():
    root = tk.Tk()
    root.title("Detección de Personas")
    root.attributes('-fullscreen', True)  # Hacer que la ventana ocupe toda la pantalla

    tab_control = ttk.Notebook(root)

    tab1 = ttk.Frame(tab_control)

    tab_control.add(tab1, text='DETECCIÓN')  # Agregar el nuevo tab para la detección de personas

    tab_control.pack(expand=1, fill='both')

    # Botón para iniciar la detección de personas
    button1 = tk.Button(tab1, text="Iniciar Detección", command=start_detection)
    button1.pack(padx=10, pady=10)

    # Reproductor de video en el tab de detección
    video_frame = tk.Frame(tab1)
    video_frame.pack(padx=10, pady=10)

    video_player = cv2.VideoCapture("video.mp4")
    video_label = tk.Label(video_frame)
    video_label.pack()

    def show_video():
        ret, frame = video_player.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (640, 480))  # Ajustar el tamaño del video si es necesario
            img = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=img)
            video_label.img = img
            video_label.config(image=img)
            video_label.after(10, show_video)  # Mostrar el siguiente fotograma
        else:
            video_player.release()
            video_label.config(image=None)

    # Cuando se selecciona el tab de detección, comenzar a mostrar el video
    tab_control.bind("<<NotebookTabChanged>>", lambda event: show_video() if tab_control.index("current") == 0 else None)

    root.mainloop()

if __name__ == "__main__":
    create_ui()
