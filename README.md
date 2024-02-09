CONTEO DE PERSONAS EN AREAS EN PYTHON:  
![image](https://github.com/christian1230/Contador-de-personas-en-reas-/assets/64487451/07c47326-c506-4f06-849e-52416286b9ba)

Este script utiliza la biblioteca tkinter para crear una interfaz de usuario simple. La detección de objetos en el video se realiza mediante YOLOv4 implementado a través de la biblioteca ultralytics. Se definen regiones de interés para el conteo, representadas por polígonos, utilizando la biblioteca shapely. Además, se emplean las bibliotecas estándar PIL para manejar imágenes, cv2 para operaciones de video y manipulación de imágenes, y numpy para operaciones numéricas eficientes. Estas dependencias son fundamentales para la ejecución del programa y se deben instalar previamente. El código también hace uso de collections.defaultdict para mantener el historial de seguimiento de cada objeto y para inicializar los contadores de las regiones de interés.
1. Definición de Regiones de Interés (ROIs):
   Se definen regiones de interés (counting_regions) utilizando polígonos. Cada región tiene un nombre, un polígono que la define, un contador inicializado en 0 y colores asociados tanto para la región como para el texto que muestra el conteo.
3. Inicio del Proceso de Detección:
   Cuando se inicia la detección, se carga el modelo YOLOv4 y se establecen las configuraciones necesarias, como los pesos del modelo, el dispositivo (CPU o GPU), y las clases de interés (en este caso, personas). Se inicia la lectura del video fotograma por fotograma utilizando OpenCV.
4. Detección de Objetos:
   Para cada fotograma, se utiliza el modelo YOLOv4 para detectar objetos, específicamente personas en este caso.
   Se realiza un seguimiento de las personas detectadas utilizando un identificador único asignado a cada una (track_id). Se registra la trayectoria de movimiento de cada persona utilizando un diccionario (track_history).
5. Conteo de Personas en Regiones de Interés:
   Para cada persona detectada, se comprueba si su posición central (centro del cuadro delimitador) está dentro de alguna de las regiones definidas. Si la persona está dentro de una región, se incrementa el contador asociado a esa región.
6. Visualización de Resultados:
   Se dibujan los polígonos que representan las regiones de interés en el fotograma, junto con el texto que muestra el conteo de personas en cada región. Se muestra el fotograma actualizado en una ventana utilizando OpenCV. El proceso continúa hasta que se alcanza el final del video o hasta que se presiona la tecla "q" para salir.
7. Interfaz de Usuario (UI):
   Se proporciona una interfaz de usuario simple utilizando la biblioteca Tkinter, que incluye un botón para iniciar la detección y un área para mostrar el video en tiempo real.
Funcionamiento
1. Interfaz: 
![image](https://github.com/christian1230/Contador-de-personas-en-reas-/assets/64487451/8c2e798a-d17d-4f6a-aac2-86da2564499c)
2. Para iniciar la detección se presiona el boton "Iniciar Detección"
![image](https://github.com/christian1230/Contador-de-personas-en-reas-/assets/64487451/999fc6db-87bc-42a7-819a-1c1b43d4ff01)
-Detección de Áreas                          
-Conteo de Personas
