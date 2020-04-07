# Basado en; “Diseño de un sistema de reconocimiento automático de matrículas de vehículos mediante una red neuronal convolucional“ realizado por Francisco José Núñez

def generar_dataset():
  X = []
  y = []
  imagen = []
  total = 0

  for ruta, subdirectorio, ficheros in os.walk('/content/drive/Team Drives/Trabajo Fin de Grado Bruno/Red Convolucion/Fnt/'):
    # Organiza los directorios donde se encuentran las letras
    subdirectorio.sort()
    # Se itera sobre cada fichero con datasets
    for nombreFichero in ficheros:
      # Se extrae la clase del nombre del fichero
      clase = nombreFichero[3:nombreFichero.index('-')]
      y.append(float(clase))
      # Se compone la ruta completa a la imagen del carácter
      rutaCompleta = os.path.join(ruta, nombreFichero)
      # Carga la imagen y la reduce a 32x32 píxeles
      imagen = io.imread(rutaCompleta,flatten=True)
      imagen_reducida = resize(imagen,(32,32))
      # Invierte la imagen
      imagen_reducida = 1 - imagen_reducida
      # Guarda la imagen de 32x32 pixeles (2D) como vector de 1024 píxeles (1D)
      X.append(imagen_reducida.reshape(1024,1))
      print (nombreFichero)
      total = total + 1

  print (total)
  # Convierte la matriz de imágenes a un array
  X = np.array(X)
  X = X.reshape(X.shape[:2])
  print (X.shape)

  from sklearn import preprocessing
  lb = preprocessing.LabelBinarizer()
  lb.fit(y)
  y = lb.transform(y)


  # Convierte el vector de clases en matriz
  y = np.array(y, dtype=float)
  # Guardar matrices como ficheros de texto
  np.savetxt('datos_x.txt', X)
  np.savetxt('datos_y.txt',y)

def cargar_dataset():
  # Comprueba que ya existen las matrices de datos
  if not(os.path.isfile('datos_x.txt')) or \
    not(os.path.isfile('datos_y.txt')):
    generar_dataset()
  X = np.loadtxt('datos_x.txt')
  y = np.loadtxt('datos_y.txt')
  print (X.shape)
  # Se generan los conjuntos de datos de entrenamiento y test
  X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.10,
  random_state=42)
  # Se divide el conjunto de entrenamiento en subconjuntos de entrenamiento y validación
  X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, test_size=0.10,
  random_state=42)
  return X_train, X_val, X_test, y_train, y_val, y_test


#Capas y nodos del entrenamiento
def peso_variable(shape,name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial,name=name)

def tendencia_variable(shape,name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial,name=name)

def convolucion2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def maxpool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

# Se resetea el grafo
tf.reset_default_graph()
# Inicia sesión de TensorFlow
sess = tf.Session()
# Marcador de posición para imágenes y clases de entrenamiento
x = tf.placeholder("float", shape=[None, 1024])
y_ = tf.placeholder("float", shape=[None, 36])

# Estructura de la red neuronal
with tf.name_scope("Reshaping_data") as scope:
  x_image = tf.reshape(x, [-1,32,32,1])

with tf.name_scope("Conv1") as scope:
  W_conv1 = peso_variable([5, 5, 1, 64],"Conv_Layer_1")
  b_conv1 = tendencia_variable([64],"Bias_Conv_Layer_1")
  h_conv1 = tf.nn.relu(convolucion2d(x_image, W_conv1) + b_conv1)
  h_pool1 = maxpool_2x2(h_conv1)

with tf.name_scope("Conv2") as scope:
  W_conv2 = peso_variable([3, 3, 64, 64],"Conv_Layer_2")
  b_conv2 = tendencia_variable([64],"Bias_Conv_Layer_2")
  h_conv2 = tf.nn.relu(convolucion2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = maxpool_2x2(h_conv2)

with tf.name_scope("Conv3") as scope:
  W_conv3 = peso_variable([3, 3, 64, 64],"Conv_Layer_3")
  b_conv3 = tendencia_variable([64],"Bias_Conv_Layer_3")
  h_conv3 = tf.nn.relu(convolucion2d(h_pool2, W_conv3) + b_conv3)
  h_pool3 = maxpool_2x2(h_conv3)

with tf.name_scope("Fully_Connected1") as scope:
  W_fc1 = peso_variable([4 * 4 * 64, 1024],"Fully_Connected_layer_1")
  b_fc1 = tendencia_variable([1024],"Bias_Fully_Connected1")
  h_pool3_flat = tf.reshape(h_pool3, [-1, 4*4*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

with tf.name_scope("Fully_Connected2") as scope:
  keep_prob = tf.placeholder("float")
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
  W_fc2 = peso_variable([1024, 36],"Fully_Connected_layer_2")
  b_fc2 = tendencia_variable([36],"Bias_Fully_Connected2")

with tf.name_scope("Final_Softmax") as scope:
  y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

with tf.name_scope("Entropy") as scope:
  cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))

with tf.name_scope("train") as scope:
  train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope("evaluating") as scope:
  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#Se guarda el entrenamiento de la iteración
saver = tf.train.Saver()
#Se inicia el grafo
sess.run(tf.initialize_all_variables())
indice = 0
#Cantidad de imágenes por iteración
lote_imagenes = 300
#Cantidad de iteraciones
epochs = 4500
all_train_accuracy = []
all_validation_accuracy = []
all_train_loss=[]

# Recuperar red previamente entrenada (si existe)
ckpt = tf.train.get_checkpoint_state("/content/drive/Team Drives/Trabajo Fin de Grado Bruno/CNN_log/")
if ckpt and ckpt.model_checkpoint_path:
  saver.restore(sess, ckpt.model_checkpoint_path)
else:
  print("No se encontro el modelo de entrenamiento. Se generara uno.")
  X_train, X_val, X_test, y_train, y_val, y_test = cargar_dataset()

  # Se centran los datos de los subconjuntos
  X_train = X_train - np.mean(X_train, axis=0)
  X_val = X_val - np.mean(X_val, axis=0)
  X_test = X_test - np.mean(X_test, axis=0)

  # Bucle de iteraciones de entrenamiento
  for i in range(epochs):
    # Carga del lote de imágenes
    lote_x = X_train[indice:indice + lote_imagenes]
    lote_y = y_train[indice:indice + lote_imagenes]
    # Se actualiza el índice
    indice = indice + lote_imagenes + 1
    if indice > X_train.shape[0]:
      indice = 0
      X_train, y_train = shuffle(X_train, y_train, random_state=0)
    if i%10 == 0:
      results_train = sess.run([accuracy,cross_entropy],feed_dict={x:lote_x,
      y_: lote_y, keep_prob: 1.0})
      train_validation = sess.run(accuracy,feed_dict={x:X_val, y_: y_val,
      keep_prob: 1.0})
      train_accuracy = results_train[0]
      train_loss = results_train[1]
      all_train_accuracy.append(train_accuracy)
      all_validation_accuracy.append(train_validation)
      all_train_loss.append(train_loss)
      print("step ",i,", training accuracy ", train_accuracy)
      print("step ",i,", validation accuracy ", train_validation)
      print("step ",i,", loss ", train_loss)
    # Guardar el modelo en cada iteración del entrenamiento
    saver.save(sess, '/content/drive/Team Drives/Trabajo Fin de Grado Bruno/CNN_log/model.ckpt', global_step=i+1)
    sess.run(train_step,feed_dict={x: lote_x, y_: lote_y, keep_prob: 0.5})

  print ("Fin del entrenamiento")
  # Para subconjuntos de entrenamiento y validación se visualiza la precisión y el error
  eje_x = np.arange(epochs/10)


  array_training = np.asanyarray(all_train_accuracy)
  array_validation = np.asanyarray(all_validation_accuracy)
  array_loss_train = np.asanyarray(all_train_loss)
  plt.figure(1)

  linea_train, = plt.plot(eje_x,array_training[0:450],label="train",linewidth=2)
  linea_test, = plt.plot(eje_x,array_validation[0:450],label="validation",linewidth=2)
  plt.legend(bbox_to_anchor=(1, 1.02), loc='upper left', ncol=1)
  plt.xlabel('epochs')
  plt.ylabel('accuracy')
  plt.show()
  plt.figure(2)
  linea_loss, = plt.plot(eje_x,array_loss_train[0:450],label="loss",linewidth=2)
  plt.legend(bbox_to_anchor=(1,1.02), loc='upper left', ncol=1)
  plt.xlabel('epochs')
  plt.ylabel('loss')
  plt.show()
  # Calcula la precisión para el subconjunto de test
  test_accuracy = sess.run( accuracy, feed_dict={x:X_test, y_: y_test, keep_prob: 1.0})
  print("Precision de test: ", test_accuracy)

#Funcion principal para reconocer el carácter
def reconocer_matricula(letras_matricula):
  matricula = ""
  clases = ["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","I","J","K","L","M","N"
  ,"O","P","Q","R","S","T","U","V","W","X","Y","Z"]
  letras_matricula = np.matrix(letras_matricula)
  classification = sess.run(y_conv, feed_dict={x:letras_matricula,keep_prob:1.0})

  for p in range(classification.shape[0]):
    pred = sess.run(tf.argmax(classification[p,:], 0))
    matricula = matricula + clases[int(pred)]
  return matricula
