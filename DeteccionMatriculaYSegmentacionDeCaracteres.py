#Función para detectar la matrícula en una imagen con camión
#Los parámetros indican lo siguiente:
####La imgOriginal es la imagen del vehículo luego de solo detectar el movimiento/La diferencia con la imagen base
####El max_height y max_width es el tamaño del kernel o recuadro que recorrerá la imagen en busca de una matrícula
####El leap es los saltos que tomara el kernel en la imagen al recorrerla en busca de una matrícula
####La variable Show es para mostrar datos extra en consola

def detectarMatricula(imgOriginal, max_height=300, max_width=500, leap=200, show=True):
  #Aplica filtro sobel para detección de bordes
  img = detectBorders(imgOriginal)

  #Obtiene las medidas de alto y ancho en la imagen
  size = getMedidas(np.uint8(img))
  height = size[0]
  width = size[1]

  min_height=0
  maxH=max_height
  min_width = 0
  maxW = max_width

  #Segmento o Kernel que recorrerá la imagen en busca de una posible matricula
  crop_img = img[min_height:max_height, min_width:max_width]

  #En esta variable estará asignada la posible matricula detectada (posición 0), junto con su valor de pixelación promedio (posición 1)
  matricula = [crop_img, sum(crop_img)/((max_height-min_height)*(max_width - min_width))]

  #Posición inicial del kernel
  pos=[0,0,0,0]

  #Recorre el ancho de la imagen
  while max_width < width:
    #Recorre el alto de la imagen
    while max_height < height:
      #Si el valor de pixelación promedio en el segmento donde está recorriendo el kernel
      #es mayor al almacenado en la variable de matrícula, lo reemplazara
      if (sum(crop_img)/((max_height-min_height)*(max_width - min_width))) > matricula[1]:
        #Si show es True, mostrará las ubicaciones donde hayan consideraciones de matrículas
        #junto con el promedio de pixelación en el área
        if show:
          print("Altura desde: ",min_height, " hasta: ", max_height)
          print("Ancho desde: ", min_width ," hasta: ", max_width)
          print("Promedio de pixelacion: ", (sum(crop_img)/(500*1100)))

        #Reemplaza la captura de consideración de matrícula obtenido por el kernel
        matricula[0] = img[min_height:max_height, min_width:max_width]
        #Reemplaza el valor promedio de pixelación obtenido por el kernel
        matricula[1] = (sum(crop_img)/((max_height-min_height)*(max_width - min_width)))

        #Guarda los cuatro puntos donde detectó posible la matrícula
        pos=[min_height,max_height, min_width,max_width]

      #Al terminar de analizar el recuadro, lo mueve acorde al valor de leap para seguir comparando
      min_height += leap
      max_height += leap
      crop_img = img[min_height:max_height, min_width:max_width]

    #Al finalizar de recorrer la columna, vuelve al inicio de esta
    #y se desplaza a la siguiente fila para continuar el recorrido
    min_height=0
    max_height=maxH
    min_width += 100
    max_width += 100
    crop_img = img[min_height:max_height, min_width:max_width]

  try:
    #Si show es True, mostrará cuánto fue el promedio de pixelación para el mejor
    #candidato de matricula, al igual que la posición donde fue capturada,
    #al igual que mostrará cómo se veía la imagen original(con detección de movimiento),
    #como se veía la imagen con el filtro, y la captura con filtro donde se
    #considera hay una matricula
    if show:
      print("MAX AVG: ", matricula[1])
      print("POSICION ", pos)

      #Todo lo asociado a la graficación de lo mencionado
      output = [imgOriginal, img, matricula[0]]
      titles = ['Original', 'Filtro Sobel', 'Intento Matricula']

      pl.figure(figsize=(15,15))

      for i in range(3):
          pl.subplot(1, 3, i+1)
          pl.imshow(output[i], cmap='gray')
          pl.title(titles[i])
          pl.xticks([])
          pl.yticks([])

      pl.show()

    #Se amplía la imagen para procesado y mejor visualización
    resize = cv2.resize(np.asarray(imgOriginal), None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC )

    #Se toma la zona donde está la posible matricula
    crop=resize[pos[0]:pos[1], pos[2]:pos[3]]

    #Se aplica blur para eliminar ruido y mejor visualización y procesado
    cropBlur = cv2.GaussianBlur(crop,(17,17),0)

    #Si show es True, mostrara las imagenes sin y con blur
    if show:
      cv2_imshow(crop)
      cv2_imshow(cropBlur)

    #Función que separa los caracteres y los almacena en un array
    caracteres = segmentarCaracteres(cropBlur)

    #Mostrará la imagen de la matrícula con un recuadro alrededor de los caracteres detectados
    if show:
      print("Matricula con caracteres detectados:")
      cv2_imshow(cropBlur)

    return caracteres

  except ValueError:
    print("No encuentra matriculas. Intenta cambiando el max_height o el max_width")
  except Exception as e:
    print(e)


#Función para separar caracteres, y marcarlos en recuadros en la imagen original
#Los parametros son:
#####croppedImage: Obtiene la captura de la posible matricula
#####show: para visualizar el procesado
def segmentarCaracteres(croppedImage, show=False):
  #En este array se guardaran los caracteres conseguidos de la matrícula
  caracteres=[]

  #Se transforma a escala de grises
  imgray = cv2.cvtColor(croppedImage,cv2.COLOR_BGR2GRAY)

  #Se binariza la imagen
  ret,thresh = cv2.threshold(imgray,127,255,0)

  #Si show es true, muestra la imagen en escala de grises y la imagen binarizada
  if show:
    print("Escala de Grises:")
    cv2_imshow(imgray)
    print("Imagen Binarizada:")
    cv2_imshow(thresh)

  #Busca contornos en la imagen y los almacena en la variable contours
  im2, contours, hierarchy = cv2.findContours(thresh,1,2)

  #Itera sobre todos los contornos detectados
  for cnt in contours:
    #Si el contorno siendo analizado tiene el área similar a la de un carácter
    #en la matricula, lo considera
    if cv2.contourArea(cnt) > 1000 and cv2.contourArea(cnt) < 2300:
      #Se obtienen todas sus medidas
      [x,y,w,h] = cv2.boundingRect(cnt)
      #Si tiene la altura similar a la de un carácter, lo tomará como valido
      if h > 35:
        #Marca el rectángulo en la imagen que venía como parámetro
        cv2.rectangle(croppedImage,(x-5,y-5),(x+w+5,y+h+5),(0,255,0),2)
        #Guarda el area donde esta el carácter
        roi = im2[y:y+h,x:x+w]
        #Lo agrega al array de caracteres (con colores invertidos, ya que así fue entrenado el sistema)
        caracteres.append(255-roi)
        #Si la variable show es True, mostrara el caracter detectado
        if show:
          cv2_imshow(255-roi)


  #Devuelve el array de los caracteres
  return caracteres
