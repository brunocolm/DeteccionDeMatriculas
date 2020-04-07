#Funcion de deteccion de movimiento. De un video se captura el momento en que ingresa
#el camión, filtra la imagen para solo mostrar este con la menor cantidad de fondo posible
#y devuelve un array con las imágenes de los camiones. Solo tiene un parámetro
######path: ruta del video
def deteccionCamiones(path):
    #Consigue las imágenes base para ser comparadas con las imágenes actuales
    #y detectar movimiento
    valores = getLightImages(path)
    base1 = valores[0]
    base2 = valores[1]

    # Función que toma el video y lo procesa
    vidObj = cv2.VideoCapture(path)

    # Contador de frames
    count = 0

    # Verifica que no hay mas frames en el video, o para finalizar el bucle
    success = 1

    #Para no mostrar contenido extra que viene en las funciones
    show = False

    #Indica si el camión ya pasó la mitad (ocupa toda la camara)
    hayCamion=False

    #Indica cuando se obtuvo una imagen candidata a que sea visible la matrícula
    foundMatricula=False

    #Cuenta los camiones que han ingresado
    camiones = 0

    #Contador para actualizar imágenes base, si no pasa un camión en 300 frames
    #se actualizarán las imágenes base
    lastRefresh = 0

    #Se almacenarán las capturas donde se encuentra el camión con la matrícula visible
    frameCamiones =[]

    #Se almacenarán las capturas donde se encuentra el camión con la matrícula visible
    #con el filtro de detección de movimiento aplicado
    frameCamionesConFiltro = []

    #Bucle que recorre los frames del video
    while success:
        #Analiza frame por frame, pasando cada uno individualmente a la variable "image"
        success, image = vidObj.read()

        #Verifica que aún hayan frames en el video, de no haberlos termina la iteración
        if image is None:
          print("Finalizo el video")
          #Si show es True, menciona la cantidad de frames totales
          if show:
            print("Cantidad de frames: ", count-1)
          #Sale de el while
          break

        #Si han pasado 300 frames desde la última actualización de imágenes base, y no se encuentra ningún camión
        #refrescará las imágenes base
        if count > lastRefresh + 300 and min(imageAvg(applyFilters(image, deteccionMovimiento(image,base1,show=show), show=show)[1]),imageAvg(applyFilters(image, deteccionMovimiento(image,base2,show=show), show=show)[1])) < 9:
          valores = getLightImages(path, startFrame = count-20)
          base1 = valores[0]
          base2 = valores[1]
          #Si show esta activo, mostrará el frame donde se actualizarán las imágenes base
          #y las nuevas imágenes base
          if show:
            print("REFRESCANDO IMAGEN BASE EN EL FRAME: ", count)
            cv2_imshow(image)
            print("NUEVA IMAGEN BASE 1:")
            cv2_imshow(base1)
            print("NUEVA IMAGEN BASE 2:")
            cv2_imshow(base2)

          #Refrescara el valor de la última vez que actualizó las imágenes base
          lastRefresh = count

        #Si hayCamion es False (No ha pasado un camión, ni esta un camión en el área de báscula)
        #y el porcentaje de movimiento considerado es mayor a un 80%, significa que un
        #camion ya pasó la mitad (ocupó gran parte del campo de visión de la cámara)
        if (not hayCamion) and min(imageAvg(applyFilters(image, deteccionMovimiento(image,base1,show=show), show=show)[1]),imageAvg(applyFilters(image, deteccionMovimiento(image,base2,show=show), show=show)[1])) > 80:
          #Cambia hayCamion a true indicando que hay un camión en el área de báscula
          hayCamion=True
          #Si show es True mostrará el frame donde el camión ingresa a la báscula
          if show:
            print("LLEGO EL CAMION EN EL FRAME: ", count)
            cv2_imshow(image)

          #Aumenta el contador de camiones 1 más
          camiones +=1

        #Si hayCamion es True(hay un camión en el área de báscula), y el porcentaje
        #que este ocupa en el campo de visión de la cámara es entre 45 y 55%,
        #detectara frames donde posiblemente la matricula este visible
        if (hayCamion and
            min(imageAvg(applyFilters(image, deteccionMovimiento(image,base1,show=show), show=show)[1]),imageAvg(applyFilters(image, deteccionMovimiento(image,base2,show=show), show=show)[1])) < 55 and
            min(imageAvg(applyFilters(image, deteccionMovimiento(image,base1,show=show), show=show)[1]),imageAvg(applyFilters(image, deteccionMovimiento(image,base2,show=show), show=show)[1])) > 45):

          #Obtiene el porcentaje considerado como ocupado por un camión
          #comparado con la imagen base 1 e imagen base 2
          mov=deteccionMovimiento(image,base1,show=show)
          deteccion1=applyFilters(image,mov,show=show)
          avg1=imageAvg(deteccion1[1])


          mov=deteccionMovimiento(image,base2,show=show)
          deteccion2=applyFilters(image,mov,show=show)
          avg2=imageAvg(deteccion2[1])

          #Se tomará la que tenga menor porcentaje considerado como ocupado por un camión

          #Caso donde el menor porcentaje lo ocupa la comparación con la imagen base 1
          if min(avg1,avg2) == avg1:
            #Si aún no ha capturado un frame donde está visible la matrícula guardará una
            if foundMatricula == False:
              #Se asigna la captura con el filtro de movimiento
              resultado = deteccion1[0]

              #En caso que show sea True, mostrará los siguientes datos:
              #La cantidad de camiones que han pasado (incluyendo el actual)
              #El número de frame donde captura la imagen
              #El porcentaje considerado como ocupado por el camión
              #La imagen donde captura el camion (sin filtro)
              #La imagen donde captura el camion (con filtro)
              if show:
                print("CAMION NUMERO: ", camiones)
                print("EN EL FRAME: ", count)
                print("PORCENTAJE DE OCUPACION: ", avg1)
                print("CAPTURA DONDE LA MATRICULA ESTA VISIBLE (SIN FILTRO):  ")
                cv2_imshow(image)
                print("CAPTURA DONDE LA MATRICULA ESTA VISIBLE (CON FILTRO):  ")
                cv2_imshow(resultado)

              #Añade al array de camiones sin filtro la captura sin filtro
              frameCamiones.append(image)
              #Añade al array de camiones con filtro la captura con filtro de movimiento
              frameCamionesConFiltro.append(resultado)
              #Cambia la variable que encontró matrícula a True
              foundMatricula= True

          #Caso donde el menor porcentaje lo ocupa la comparación con la imagen base 1
          elif min(avg1,avg2) == avg2:
            #Si aún no ha capturado un frame donde está visible la matrícula guardará una
            if foundMatricula== False:
              #Se asigna la captura con el filtro de movimiento
              resultado = deteccion2[0]

              #En caso que show sea True, mostrará los siguientes datos:
              #La cantidad de camiones que han pasado (incluyendo el actual)
              #El número de frame donde captura la imagen
              #El porcentaje considerado como ocupado por el camión
              #La imagen donde captura el camion (sin filtro)
              #La imagen donde captura el camion (con filtro)
              if show:
                print("CAMION NUMERO: ", camiones)
                print("EN EL FRAME: ", count)
                print("PORCENTAJE DE OCUPACION: ", avg1)
                print("CAPTURA DONDE LA MATRICULA ESTA VISIBLE (SIN FILTRO):  ")
                cv2_imshow(image)
                print("CAPTURA DONDE LA MATRICULA ESTA VISIBLE (CON FILTRO):  ")
                cv2_imshow(resultado)

              #Añade al array de camiones sin filtro la captura sin filtro
              frameCamiones.append(image)
              #Añade al array de camiones con filtro la captura con filtro de movimiento
              frameCamionesConFiltro.append(resultado)
              #Cambia la variable que encontró matrícula a True
              foundMatricula = True

        #En caso que haya salido un camión, se refrescaran los valores y se encontrarán nuevas imágenes base
        #Se determina que un camión sale al tener hayCamion en True (entro un camion)
        #Y la ocupación de píxeles del movimiento (comparado con las imágenes base) es menor a 9%
        if hayCamion and min(imageAvg(applyFilters(image, deteccionMovimiento(image,base1,show=show), show=show)[1]),imageAvg(applyFilters(image, deteccionMovimiento(image,base2,show=show), show=show)[1])) < 9:
          #Reiniciará las variables que indican que pasó un camión y que obtuvo una captura a False a la espera de otro camión
          hayCamion = False
          foundMatricula = False

          #Obtiene nuevas imágenes base, consideradas poco después que salió el camion
          valores = getLightImages(path, startFrame = count+10)
          base1 = valores[0]
          base2 = valores[1]

          #Guarda el último momento que se actualizaron las imágenes base
          lastRefresh = count

          #Si show es true, mostrará el frame donde sale el camión, y las nuevas imágenes base
          if show:
            print("SE FUE EL CAMION EN EL FRAME: ", count)
            cv2_imshow(image)
            print("NUEVA IMAGEN BASE 1:")
            cv2_imshow(base1)
            print("NUEVA IMAGEN BASE 2:")
            cv2_imshow(base2)

        #Contador de frames
        count += 1

    #Al finalizar la ejecución, si show es True mencionara la cantidad de camiones
    #encontrados y los mostrará con y sin filtro
    if show:
      print("CAMIONES ENCONTRADOS: ", camiones)
      print("CAMIONES SIN FILTRO:")
      for camion in frameCamiones:
        cv2_imshow(camion)
      print("CAMIONES CON FILTRO DE MOVIMIENTO:")
      for camion in frameCamionesConFiltro:
        cv2_imshow(camion)

    #Devolverá el array de camiones con filtro
    return frameCamionesConFiltro
