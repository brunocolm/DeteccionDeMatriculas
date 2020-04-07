#Función que captura el porcentaje de movimiento de cada frame
#siendo comparado con las imágenes base. Los parametros son:
#####path: ruta del video donde se quieren sacar los frames
#####initialFrame: en caso de solo querer el gráfico para un trozo del video
#####              se debe especificar el frame inicial
#####lastFrame: en caso de solo querer el gráfico para un trozo del video
#####              se debe especificar el frame final
#De querer todo el video, los parametros seran
#respectivamente 0 y 99999 (valor improbable que alcance)
#####show=Muestra información extra en la consola, como la imagen solo del movimiento,
#####     la mascara de esta, y el porcentaje de movimiento considerado

def ValueCapture(path, initialFrame=0, lastFrame=99999, show=False):
  #Consigue las imágenes base para comparar en la detección de movimiento
    valores = getLightImages(path)
    base1 = valores[0]
    base2 = valores[1]

    # Función que toma el video y lo procesa
    vidObj = cv2.VideoCapture(path)

    # Contador de frames
    count = 0

    # Verifica que no hay mas frames en el video, o para finalizar el bucle
    success = 1

    #En esta variable irán arrays donde en la posición 0 está el frame capturado
    #y en la posición 1 está el porcentaje de movimiento considerado
    valores=[]

    #Bucle que itera sobre los frames
    while success:
        #Analiza frame por frame, pasándolas individualmente a la variable "image"
        success, image = vidObj.read()

        #Si no hay mas frames en el video finaliza el bucle
        if image is None:
          break

        #Considera a partir del frame inicial indicado
        #(o en su defecto desde el inicio del video)
        if count >initialFrame:
          #Aplica los procesados necesarios (detección de movimiento y los filtros)
          #para eliminar el ruido
          deteccion1=deteccionMovimiento(image,base1, show=False)
          filtro1 = applyFilters(image,deteccion1, show=False)
          mov1 = filtro1[0]
          binarizado1=filtro1[1]

          deteccion2=deteccionMovimiento(image,base2, show=False)
          filtro2 = applyFilters(image,deteccion2, show=False)
          mov2 = filtro2[0]
          binarizado2=filtro2[1]

          #Consigue el porcentaje de movimiento considerado
          #(todo lo que en la captura no es es parte de la imagen base)
          avg1 = (sum(binarizado1)/binarizado1.size)
          avg2 = (sum(binarizado2)/binarizado2.size)

          #El valor mínimo será considerado el porcentaje de movimiento
          #para el frame actual
          valor = min(avg1,avg2)
          if show:
            print("Frame: ", count)
            print("Porcentaje de movimiento: ", valor)
            if min(avg1,avg2)== avg1:
              print("Deteccion de movimiento:")
              cv2_imshow(mov1)
              print("Mascara de movimiento:")
              cv2_imshow(binarizado1)
            else:
              print("Deteccion de movimiento:")
              cv2_imshow(mov2)
              print("Mascara de movimiento:")
              cv2_imshow(binarizado2)

          #Guarda en el array de valores el porcentaje de movimiento
          #considerado en el frame actual
          valores.append(valor*100/255)

        #Si llega al frame indicado como final terminara la iteración.
        #Ante la ausencia de este parámetro terminará en el frame 99999
        #o al finalizar el video
        if count == lastFrame:
          success = 0

        #Aumenta el contador de frames
        count +=1

    #Devuelve el array con los valores de movimiento considerados
    return valores

#Función que genera el gráfico de movimiento. Los parametros son:
#####porcentajeMovimientos: array que contiene el porcentaje de movimiento
#####                       calculado de cada frame
#####width: ancho que tendrá la gráfica
#####height: alto que tendra la grafica
#Esas dimensiones de gráficas son óptimas para una captura de aproximadamente
#1500 frames (En este caso seria un video de aproximadamente 2min 30segs)
def generarGraficoMovimiento(porcentajeMovimientos, width=15,height=2):
  #Asignación del gráfico con sus características
  fig, axs = plt.subplots(1,1, figsize=(width,height))

  #Ingresar el array de datos
  axs.plot(porcentajeMovimientos)

  #Características del gráfico
  axs.set_title("Deteccion de Movimiento")
  axs.set_xlabel("Frame")
  axs.set_xlim(left=0)
  axs.set_ylabel("% Ocupado")

  #Muestra cada cuantos números muestra el valor en el eje X
  spacing = 100
  majorLocator = MultipleLocator(spacing)
  axs.xaxis.set_major_locator(majorLocator)

  #Crea un grid en el gráfico
  axs.grid(True, which='major')


#Para generar un gráfico se llamaría de la siguiente manera:
#val = ValueCapture(rutaVideo)
#generarGraficoMovimiento(val)
