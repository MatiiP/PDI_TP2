import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


# Rutas
resistencias_path = './Resistencias'
output_path = './Resistencias_OUT'
os.makedirs(output_path, exist_ok=True)

# Lista completa de imágenes
imagenes = [f"R{i}_{letra}" for i in range(1, 11) for letra in "abcd"]

# Función para visualizar los pasos del procesamiento
def visualizar_pasos(original, mascara, resultado, titulo="Transformada"):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mascara, cmap="gray")
    plt.title("Máscara Azul")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB))
    plt.title(titulo)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
# Función para procesar cada imagen
def procesar_imagen(nombre_imagen):
    print(f"\nProcesando: {nombre_imagen}.jpg")

    path_img = os.path.join(resistencias_path, f"{nombre_imagen}.jpg")
    img = cv2.imread(path_img)
    if img is None:
        print(f"  No se pudo leer {path_img}")
        return

    # Máscara azul (ajustada)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    azul_bajo = np.array([90, 30, 20])
    azul_alto = np.array([140, 255, 255])
    mascara = cv2.inRange(hsv, azul_bajo, azul_alto)

    # Suavizado morfológico
    kernel = np.ones((3, 3), np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)

    # Buscar contornos
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        print("  No se detectó contorno azul.")
        return

    contorno_mayor = max(contornos, key=cv2.contourArea)

    # Rectángulo rotado
    rect_rotado = cv2.minAreaRect(contorno_mayor)
    box = cv2.boxPoints(rect_rotado)
    box = np.array(box, dtype="float32")

    # Ordenar esquinas
    s = box.sum(axis=1)
    diff = np.diff(box, axis=1)
    rect = np.zeros((4, 2), dtype="float32")
    rect[0] = box[np.argmin(s)]      # top-left
    rect[2] = box[np.argmax(s)]      # bottom-right
    rect[1] = box[np.argmin(diff)]   # top-right
    rect[3] = box[np.argmax(diff)]   # bottom-left

    ancho = int(max(
        np.linalg.norm(rect[1] - rect[0]),
        np.linalg.norm(rect[2] - rect[3])
    ))
    alto = int(max(
        np.linalg.norm(rect[3] - rect[0]),
        np.linalg.norm(rect[2] - rect[1])
    ))

    dst = np.array([
        [0, 0],
        [ancho, 0],
        [ancho, alto],
        [0, alto]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    img_transformada = cv2.warpPerspective(img, M, (ancho, alto))

    # Asegurar orientación horizontal
    if img_transformada.shape[0] > img_transformada.shape[1]:
        img_transformada = cv2.rotate(img_transformada, cv2.ROTATE_90_CLOCKWISE)

    # Guardar
    salida_path = os.path.join(output_path, f"{nombre_imagen}_out.jpg")
    cv2.imwrite(salida_path, img_transformada)
    print(f"  Guardado en: {salida_path}")

    # Mostrar pasos
    visualizar_pasos(img, mascara, img_transformada)

# Procesar todas las imágenes
for nombre in imagenes:
    procesar_imagen(nombre)

##### Codigo Preliminar para detectar colores de franjas en resistencias

#### Todavia no pudimos lograr una deteccion clara de estos colores en dichas resistencias.

# Ruta a la carpeta que contiene las imágenes
# Crea carpeta de salida si no existe
os.makedirs("Resultados", exist_ok=True)

# Definir función para clasificar colores
def get_color_name(h, s, v):
    if v < 40:
        return "Negro"
    elif s < 40 and v > 200:
        return "Blanco"
    elif 0 <= h <= 10 or 160 <= h <= 179:
        return "Rojo"
    elif 11 <= h <= 25:
        return "Naranja"
    elif 26 <= h <= 34:
        return "Amarillo"
    elif 35 <= h <= 85:
        return "Verde"
    elif 86 <= h <= 125:
        return "Azul"
    elif 126 <= h <= 145:
        return "Violeta"
    elif 10 <= h <= 25 and s < 150 and v < 150:
        return "Marrón"
    else:
        return "Desconocido"

# Recorrer imágenes R1_a_out.jpg a R10_a_out.jpg
for i in range(1, 11):
    nombre_img = f"R{i}_a_out.jpg"
    ruta = os.path.join("Resistencias_OUT", nombre_img)

    img = cv2.imread(ruta)
    if img is None:
        print(f" No se encontró la imagen: {ruta}")
        continue

    # Convertir a gris y binarizar para obtener resistencia
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 10), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=5)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, 255, -1)
    resistor_only = cv2.bitwise_and(img, img, mask=mask)

    # Canny y cierre morfológico para franjas
    blurred = cv2.GaussianBlur(resistor_only, (7, 7), 0)
    edges = cv2.Canny(blurred, 30, 80)
    kernel_franjas = np.ones((8, 6), np.uint8)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_franjas, iterations=2)
    franjas_contours, _ = cv2.findContours(closed_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Analizar franjas
    color_franjas = []
    for cnt in franjas_contours:
        area = cv2.contourArea(cnt)
        if area < 100 or area > 3000:
            continue

        mask_franja = np.zeros_like(gray)
        cv2.drawContours(mask_franja, [cnt], -1, 255, -1)
        franja_region = cv2.bitwise_and(resistor_only, resistor_only, mask=mask_franja)
        pixels = franja_region[mask_franja == 255]

        if len(pixels) == 0:
            continue

        mean_color_bgr = np.mean(pixels, axis=0).astype(np.uint8)
        mean_color_hsv = cv2.cvtColor(np.uint8([[mean_color_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        color_name = get_color_name(*mean_color_hsv)

        if color_name in ["Negro", "Marrón", "Rojo", "Naranja", "Amarillo", "Verde", "Azul", "Violeta"]:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                color_franjas.append((color_name, cx))

    # Ordenar horizontalmente
    color_franjas.sort(key=lambda x: x[1])
    detected_colors = [c[0] for c in color_franjas[:4]]

    print(f" [{nombre_img}] Colores detectados:", detected_colors)

    # Dibujar sobre copia
    img_resultado = img.copy()
    for color_name, cx in color_franjas[:4]:
        cy = img.shape[0] // 2
        cv2.putText(img_resultado, color_name, (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.line(img_resultado, (cx, 0), (cx, img.shape[0]), (0, 255, 0), 1)

    # Mostrar imagen con resultado
    plt.figure(figsize=(10, 3))
    plt.imshow(cv2.cvtColor(img_resultado, cv2.COLOR_BGR2RGB))
    plt.title(f"{nombre_img} - Franjas detectadas")
    plt.axis("off")
    plt.show()

    # Guardar imagen procesada
    cv2.imwrite(os.path.join("Resultados", f"resultado_{nombre_img}"), img_resultado)