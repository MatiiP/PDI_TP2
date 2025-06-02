import cv2
import numpy as np
import matplotlib.pyplot as plt

imagen = cv2.imread('./placa.png')
imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

plt.imshow(imagen_rgb)
plt.title('Placa') 
plt.axis('off') 
plt.show()

# -------------------------------------------

 # Convertir a escala de grises

gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
plt.imshow(gris, cmap='gray')
plt.axis('off') 
plt.show()


# -------------------------------------------

# Filtro Gausseano para suavisar bordes

f_blur = cv2.GaussianBlur(gris,(11,11),1)
plt.imshow(f_blur, cmap='gray')
plt.axis('off')
plt.show()

# -------------------------------------------

# Canny

canny = cv2.Canny(f_blur, 50, 80)
plt.imshow(canny, cmap='gray')
plt.axis('off')
plt.show()

# -------------------------------------------

# Kernel con estructura rectangular
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))

# Aplicamos clausura (closing): dilatación seguida de erosión
canny_closed = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)

# Mostrar resultado
plt.imshow(canny_closed, cmap='gray')
plt.axis('off')
plt.title('Canny + Clausura')
plt.show()

# -------------------------------------------

# Dilatación de las lineas

# Kernel para engrosar las líneas
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))  

# Dilatación
canny_dilated = cv2.dilate(canny_closed, kernel, iterations=1)

plt.figure(figsize=(8, 6))
plt.imshow(canny_dilated, cmap='gray')
plt.title('Canny Engrosado (Dilatación)')
plt.axis('off')
plt.show()

# -------------------------------------------

# Componentes conectadas

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(canny_dilated, connectivity=8)

print(f"Número total de componentes conectadas: {num_labels - 1}")

# Visualizar todas las componentes con colores
labels_colored = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
colors[0] = [0, 0, 0]  

for i in range(num_labels):
    labels_colored[labels == i] = colors[i]

plt.figure(figsize=(12, 6))
plt.imshow(labels_colored)
plt.title(f'Todas las Componentes Conectadas ({num_labels-1} componentes)')
plt.axis('off')
plt.show()

#---------------------------------------------------------

imagen_con_rectangulos = imagen_rgb.copy()

n_total = 0

resistencias = []

for i in range(1, num_labels):
    x, y, w, h, area = stats[i]
    if area > 3200:
        aspect_ratio = max(w,h) / min(w,h)
        print(aspect_ratio)
        if aspect_ratio < 1.7:
            submask = imagen_rgb[y:y+h, x:x+w]
            gris_submask = cv2.cvtColor(submask, cv2.COLOR_BGR2GRAY)
            gris_submask = cv2.GaussianBlur(gris_submask, (9,9),0)
            circulos = cv2.HoughCircles(gris_submask, cv2.HOUGH_GRADIENT, dp=1, minDist=250, param1=70, param2=50, minRadius=25, maxRadius=200)

            if circulos is not None:
                circulos = np.round(circulos[0, :]).astype("int")
                
                # Define radius thresholds for circle sizes
                small_threshold = 100
                medium_threshold = 150
                
                # Group circles by size
                small_circles = []
                medium_circles = []
                large_circles = []
                
                for circle in circulos:
                    x_c, y_c, r = circle
                    if r < small_threshold:
                        small_circles.append(circle)
                    elif r < medium_threshold:
                        medium_circles.append(circle)
                    else:
                        large_circles.append(circle)
                
                # Process each circle group
                for circle in circulos:
                    x_c, y_c, r = circle
                    
                    # Create a circular mask
                    mask = np.zeros(gris_submask.shape[:2], dtype=np.uint8)
                    cv2.circle(mask, (x_c, y_c), r-5, 255, -1)  # -5 to avoid border pixels
                    
                    # Calculate brightness of circle interior
                    circle_pixels = gris_submask[mask == 255]
                    mean_value = np.mean(circle_pixels)
                    
                    # Check if circle interior is bright enough (white)
                    brightness_threshold = 130
                    
                    if mean_value > brightness_threshold:
                        # Add the offset of the submask position to get global coordinates
                        global_x = x + x_c
                        global_y = y + y_c
                        
                        # Draw circles with different colors based on size
                        if r < small_threshold:
                            color = (255, 0, 0)  # Red for small circles
                        elif r < medium_threshold:
                            color = (0, 255, 0)  # Green for medium circles
                        else:
                            color = (0, 0, 255)  # Blue for large circles
                            
                        cv2.circle(imagen_con_rectangulos, (global_x, global_y), r, color, 6)
        elif area > 10000:
            color = (255, 0, 255)
            cv2.rectangle(imagen_con_rectangulos, (x, y), (x + w, y + h), color, 6)
            cv2.putText(imagen_con_rectangulos, 'CHIP', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 5, color, 2)
        else:
            submask = imagen_rgb[y:y+h, x:x+w]

            # Color objetivo en BGR (ajusta según tu caso)
            color_objetivo = np.array([200, 160, 100], dtype=np.uint8)

            # Umbral de diferencia permitida (ajustable)
            umbral = 50

            # Crear máscara de píxeles que están dentro del rango de color
            lower = np.clip(color_objetivo - umbral, 0, 255)
            upper = np.clip(color_objetivo + umbral, 0, 255)

            # Aplicar la detección de color
            mask = cv2.inRange(submask, lower, upper)

            # Número total de píxeles en la submáscara
            total_pixeles = submask.size // 3  # dividimos entre 3 porque son 3 canales (BGR)

            # Número de píxeles que coinciden con el color objetivo
            pixeles_coincidentes = np.count_nonzero(mask)  # cuenta los píxeles blancos (255)

            # Porcentaje
            porcentaje = (pixeles_coincidentes / total_pixeles) * 100

            print(f"Porcentaje de color objetivo en la submáscara: {porcentaje:.2f}%")
            if porcentaje > 23:
                color = (0, 255, 255)
                cv2.rectangle(imagen_con_rectangulos, (x, y), (x + w, y + h), color, 6)
                n_total += 1
                resistencias.append(i)

plt.figure(figsize=(12, 6))
plt.imshow(imagen_con_rectangulos)
plt.title(f'Resistencias Encontradas: ({n_total} resistencias)')
plt.axis('off')
plt.show()

print(resistencias)


