#En la variable "resultado" se almacenarán arrays donde la primera posición es el contador de matrículas, comenzando por 1
#y la segunda posición es un array que contiene:
#Arrays donde la primera posición es la imagen del carácter y la segunda posición el reconocimiento de dicho carácter
resultado = []
#contador
i=1
for matricula in arrayCamiones:
  #Array donde iran las detecciones de una misma matrícula
  deteccion = []
  #Itera cada carácter para el reconocimiento
  for caracter in detectarMatricula(matricula,show=False):
    #Redimensiona el carácter al tamaño aceptado por la red entrenada
    caracter = cv2.resize(caracter, (32,32))
    #Lo transforma a un vector de 1D
    caracter_1d=caracter.flatten()
    #Guarda el reconocimiento en una variable
    reconocimiento = reconocer_matricula(caracter_1d)
    #Almacena en detección la información del carácter
    deteccion.append([caracter, reconocimiento])
  #Almacena en resultado toda la información de las matrículas y las detecciones
  resultado.append([i, deteccion])
  #Aumenta el contador de matrículas
  i +=1


#Itera sobre todas las matrículas
for matriculaN in resultado:
  print("Matricula N: ", matriculaN[0])

  #Si no se detectaron caracteres en la matrícula lo mostrara
  if len(matriculaN[1]) == 0:
    print("No se detectaron caracteres en esta matricula")
  #Muestra la imagen del carácter con el reconocimiento deducido por la red
  else:
    for caracter in matriculaN[1]:
      cv2_imshow(caracter[0])
      print("reconocimiento: ", caracter[1], "\n")
