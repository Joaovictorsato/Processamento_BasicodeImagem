## 1. Realce e Ajuste de Intensidade
### Filtro: Equalização de Histograma (YUV)
- A equalização de histograma foi aplicada ao canal de luminância (Y) da imagem convertida para o espaço de cor YUV.
- Esse processo melhora o contraste da imagem, distribuindo de forma mais uniforme os valores de intensidade.

**Código:**
```python
def intensity_adjustment(image):
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
```

---

## 2. Redução de Ruído e Suavização
### Filtro: Filtro Bilateral
- O filtro bilateral foi utilizado para reduzir o ruído preservando as bordas da imagem. Ele funciona realizando uma média ponderada dos pixels vizinhos, considerando tanto a proximidade espacial quanto a similaridade de intensidade.

**Código:**
```python
def noise_reduction(image):
    return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
```

---

## 3. Detecção de Bordas
### Filtro: Algoritmo de Canny
- O algoritmo de Canny foi usado para detectar bordas na imagem. Ele aplica um suavizador Gaussian antes de calcular gradientes de intensidade para identificar bordas com base em limites superiores e inferiores definidos.

**Código:**
```python
def edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    return edges
```

---

## 4. Detecção de Formas e Texturas
### Filtro: Detecção de Contornos
- A detecção de contornos foi realizada convertendo a imagem para tons de cinza, aplicando uma binarização (limiarização) simples e depois utilizando o método `cv2.findContours` para encontrar os contornos na imagem.

**Código:**
```python
def shape_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    return contour_image
```

---

## 5. Transformações Geométricas
### Filtro: Rotação
- Uma transformação geométrica foi aplicada para rotacionar a imagem em 45 graus. A matriz de rotação foi calculada com o centro da imagem como ponto fixo.

**Código:**
```python
def geometric_transformation(image):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle=45, scale=1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h))
    return rotated
```

---

## 6. Filtros Morfológicos
### Filtro: Dilatação
- O filtro de dilatação foi aplicado para expandir as regiões brancas (mais brilhantes) de uma imagem em tons de cinza. Um kernel de 5x5 foi usado para a operação.

**Código:**
```python
def morphological_filter(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(gray, kernel, iterations=1)
    return dilated
```

---

## Requisitos
- **Bibliotecas:** Certifique-se de ter as bibliotecas `opencv-python`, `matplotlib` e `numpy` instaladas.
  - Para instalar, utilize:
    ```bash
    pip install opencv-python matplotlib numpy
    ```

## Execução
- Substitua o caminho da imagem no script pelo caminho da sua imagem ou use a imagem `lena.jpg`.
- Execute o script e visualize os resultados de cada filtro em janelas ou gráficos exibidos.

