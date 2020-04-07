#Importación de librerías
from pylab import *
from numpy import *
import matplotlib.pyplot as pl
import cv2
from scipy.ndimage import filters
import PIL
from google.colab import drive
from IPython.display import display

# Autoriza la conexion con google drive
drive.mount('/content/drive')

#Funciones auxiliares esenciales en el código

#Ya que google colab no permite la función cv2.imshow(), ofrecen una alternativa
#a traves de esta funcion
def cv2_imshow(a):
  a = a.clip(0, 255).astype('uint8')
  # cv2 stores colors as BGR; convert to RGB
  if a.ndim == 3:
    if a.shape[2] == 4:
      a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
    else:
      a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
  display(PIL.Image.fromarray(a))

#Obtiene el alto y ancho de la imagen
#El parámetro img es la imagen a analizar
def getMedidas(img):
  if len(img.shape) == 3:
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  height, width = (array(img)).shape
  return [height, width]

#Obtiene las "imágenes base" con la cual se hará el algoritmo de detección de movimiento
#Comparará las imagenes actuales con las imágenes resultantes de esta función
#Los parámetros representan lo siguiente:
#####path: Ruta de donde se conseguirán dichas imágenes base
#####maxFrames: cantidad de frames a considerar para conseguir las imágenes base
#####startFrame: frame inicial desde donde empezará a considerar imagenes base
#####show: muestra datos extra como el histograma de las imágenes base
def getLightImages(path, maxFrames=20, startFrame = 0, show=False):
    # Función que toma el video y lo procesa
    vidObj = cv2.VideoCapture(path)

    # Contador de frames
    count = 0

    # Verifica que no hay mas frames en el video, o para finalizar el bucle
    success = 1

    while success:
        #Analiza frame por frame, pasándolas individualmente a la variable "image"
        success, image = vidObj.read()

        #Toma las medidas de la imagen, y determina el valor promedio de pixelación
        medidas = getMedidas(image)
        height = medidas[0]
        width = medidas[1]
        avg = (sum(image)/(height*width))

        #Solo se considerara a partir del frame indicado
        if count < startFrame:
          pass

        #Una vez que llega al frame indicado, tomara los primeros dos, los ordenara
        #por intensidad de pixelación, y asignará a las variables array
        #"lowValue" y "highValue" la imagen analizada en su primera posición y
        #el valor de pixelación promedio en la segunda posición
        elif count == startFrame:
          firstFrame = [image, avg]
        elif count == startFrame + 1:
          if firstFrame[1] > avg:
            highValue = firstFrame
            lowValue = [image, avg]
          else:
            lowValue = firstFrame
            highValue = [image, avg]

        #El resto de los frames serán comparados con los dos almacenados anteriormente,
        #si encuentra un valor menor que "lowValue" lo reemplazará
        #si encuentra un valor mayor que "highValue" lo reemplazará
        elif count > startFrame + 1 and count < startFrame + maxFrames:
          if avg > highValue[1]:
            highValue = [image, avg]
          elif avg < lowValue[1]:
            lowValue = [image, avg]

        #Finalmente, al ver la cantidad de frames indicada, devolvera
        #las imágenes guardadas en las dos variables. La de menores valores
        #de pixelacion, y la de mayores valores de pixelación
        else:
          #Si la variable show es True, mostrará la suma de todos sus píxeles
          #junto con el histograma de ambas imágenes
          if show:
            print("La suma de los pixeles para la imagen base 1 es: ", sum(lowValue[0].flatten()))
            cv2_imshow(lowValue[0])
            # Crear el histograma
            figure()
            fig = pl.hist(lowValue[0].flatten(),256)
            # 256 es el número de divisiones del histograma
            pl.title('Histograma Base 1')
            pl.xlabel("Valor")
            pl.ylabel("Frecuencia")
            pl.grid(None)
            pl.show()

            print("La suma de los pixeles para la imagen base 2 es: ", sum(highValue[0].flatten()))
            cv2_imshow(highValue[0])
            # Crear el histograma
            figure()
            fig = pl.hist(highValue[0].flatten(),256)
            # 256 es el número de divisiones del histograma
            pl.title('Histograma base 2')
            pl.xlabel("Valor")
            pl.ylabel("Frecuencia")
            pl.grid(None)
            pl.show()
          return [lowValue[0], highValue[0]]

          #Sale de la iteración
          success = 0

        #Continúa al siguiente frame
        count +=1

#Consigue el valor de píxeles promedio en una imagen entre 0 y 100
#El parámetro img es la imagen a analizar
def imageAvg(img):
  #En caso que la imagen no sea en escala de grises, la convertira
  if len(img.shape) == 3:
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  height, width = getMedidas(img)

  #Devuelve el promedio calculado de la siguiente manera:
  #La suma de sus píxeles, entre su alto*ancho
  #Esto dará un valor del 0 al 255, al dividirlo entre 255 dará un valor entre 0 y 1
  #Luego se multiplicará por 100 para tener el valor entre 0 y 100
  return (sum(img)*100/(height*width*255))

#Función que detecta movimiento a partir de dos imágenes, devolverá la imagen
#Los parámetros son los siguientes:
#####img1: primera imagen a comparar
#####img2: segunda imagen con la cual se comparará
#####show: mostrar la imagen en la consola, para analisis
def deteccionMovimiento(img1,img2, show=False):
  #Convierte ambas imágenes a escala de grises
  img1=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
  img2=cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

  #Aplica la diferencia entre ambas imágenes para detectar el movimiento
  dif = cv2.absdiff(img1, img2)

  #Muestra la diferencia de las imágenes
  if show:
    cv2_imshow(dif)

  #Devuelve la imagen luego de calcular la diferencia
  return dif

#Función donde muestra la porción de la imagen original que supone movimiento
#Los parametros de entrada son:
#####img: imagen original. Se le aplicara un filtro para mostrar solo el área con movimiento
#####deteccionMov: imagen resultante de la funcion deteccionMovimiento.
#####th: valor en el cual será binarizado. Todos los pixeles con valor menor a este serán 0(negro), y los mayores serán 255(blanco)
#####show: muestra las imágenes en consola para análisis.
def applyFilters(img, deteccionMov,th=40, show=False):

  #Se binariza la imagen siendo el punto de corte el valor de th
  ret,binarizado = cv2.threshold(deteccionMov,th,255,cv2.THRESH_BINARY)

  #Se crea un kernel que recorre la matriz aplicando filtros
  kernel = np.ones((12,12),np.uint8)

  #Se itera el efecto de dilatación, ampliando la zona con valores de 255 (blancos)
  #Se hace 3 veces para tener un mejor resultado
  dil = cv2.dilate(binarizado,kernel,iterations = 3)

  #Se itera el efecto de erosión, ampliando la zona con valores de 0 (negros)
  #Se hace 3 veces para tener un mejor resultado
  ero = cv2.erode(dil,kernel,iterations = 3)
  mask = ero

  #Se realizan ambos efectos para eliminar la mayor cantidad de ruido posible

  #Se mezcla la imagen original con la imagen binarizada luego de los filtros
  #Esto mostrará en la imagen original solo las áreas donde en la binarizada hay blancos
  #mostrando solo lo considerado movimiento
  res = cv2.bitwise_and(img,img, mask= mask)

  #Si se quieren ver más detalles, al tener la variable show en True mostrara
  #la imagen luego de cada paso
  if show:
    cv2_imshow(binarizado)
    cv2_imshow(dil)
    cv2_imshow(ero)
    cv2_imshow(res)

  #Devolverá un array con la imagen actual con el filtro aplicado
  #y el filtro solo
  return [res, mask]

#Función utilizada para la detección de la matrícula en la captura donde está visible
#Los parametros son:
#####img: La imagen a la cual se le aplicarán los filtros
#####show: Para mostrar mas information.
def detectBorders(img, show=False):
  #Se aumenta el tamaño para hacer efectos mejores adaptado
  resize = cv2.resize(np.asarray(img), None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC )

  #Filtro para aplicar blur y facilitar la detección de bordes evitando ruido
  gaussian = cv2.GaussianBlur(resize, (51, 51), 0)

  # Detección de bordes mediante filtro Sobel
  imx = zeros(gaussian.shape)
  imx=filters.sobel(gaussian,1,imx)
  imy = zeros(gaussian.shape)
  imy=filters.sobel(gaussian,0,imy)
  im_sobel = sqrt(imx**2+imy**2)

  #Para visualizar, los muestra en colores invertidos por lo cual al invertirlos nuevamente
  #mostrará los colores originales
  inverso =255-im_sobel

  #Si show es true, mostrará el resultado de la detección de bordes
  if show:
    cv2_imshow(im_sobel)

  #Devolverá la imagen con los filtros aplicados
  return im_sobel
