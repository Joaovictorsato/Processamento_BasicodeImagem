import cv2
import numpy as np
from matplotlib import pyplot as plt

# Função para exibir a imagem
def show_image(title, image):
    plt.figure(figsize=(6, 6))
    plt.title(title)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Carregar a imagem
image = cv2.imread('lena.jpg')  # Substitua pelo caminho da sua imagem
if image is None:
    raise FileNotFoundError("Imagem não encontrada. Certifique-se de especificar o caminho correto.")

# 1. Realce e Ajuste de Intensidade (Equalização de histograma - YUV)
def intensity_adjustment(image):
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

adjusted_image = intensity_adjustment(image)
show_image("Realce e Ajuste de Intensidade", adjusted_image)

# 2. Redução de Ruído e Suavização (Filtro bilateral)
def noise_reduction(image):
    return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

smoothed_image = noise_reduction(image)
show_image("Redução de Ruído e Suavização", smoothed_image)

# 3. Detecção de Bordas (Canny)
def edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    return edges

edges_image = edge_detection(image)
plt.figure(figsize=(6, 6))
plt.title("Detecção de Bordas")
plt.imshow(edges_image, cmap='gray')
plt.axis('off')
plt.show()

# 4. Detecção de Formas e Texturas (Detecção de contornos)
def shape_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    return contour_image

shapes_image = shape_detection(image)
show_image("Detecção de Formas e Texturas", shapes_image)

# 5. Transformações Geométricas (Rotação)
def geometric_transformation(image):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle=45, scale=1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h))
    return rotated

rotated_image = geometric_transformation(image)
show_image("Transformações Geométricas", rotated_image)

# 6. Filtros Morfológicos (Dilatação)
def morphological_filter(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(gray, kernel, iterations=1)
    return dilated

dilated_image = morphological_filter(image)
plt.figure(figsize=(6, 6))
plt.title("Filtros Morfológicos")
plt.imshow(dilated_image, cmap='gray')
plt.axis('off')
plt.show()