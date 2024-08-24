import tensorflow as tf
import numpy as np
#CODIGO ABIERTO. HACED LO QUE QUERAIS CON ESTO

#comprobar si la gpu es detectada #GPU AMD NO ES COMPATIBLE
print(tf.config.list_physical_devices('GPU'))

celsius = np.array([-40,-10,0,8,15,22,38], dtype=float)
fahrenheit = np.array([-40,14,32,46,59,72,100], dtype=float)

capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo = tf.keras.Sequential([capa])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1), #adam le dice a la red como ajustar los pesos y sesgos de manera eficiente
    loss='mean_squared_error' #mean squared error considera que una pequeña cantidad de errores grandes es PEOR que una gran cantidad de errores pequeños
)

print("[-------!]Comenzando entrenamiento...")
historial = modelo.fit(celsius, fahrenheit, epochs=1000, verbose=False) #epochs singifica el numero de vueltas que le va a dar al problema. verbose en false impide que el LOG de cada una de las vueltas se muestre por terminal
print('[-------!]modelo entrenado!')

# import matplotlib.pyplot as plt #el codigo comentado solo funciona en google colab
# plt.xlabel("# Epoca")
# plt.ylabel("Magnitud de perdida")
# plt.plot(historial.history["loss"])
# plt.show
print('[-------!]hagamos una prediccion')
resultado = modelo.predict(np.array([100.0])) #100 son los grados celsius que va a intentar predecir su valor en fahrenheit
print('[-------!]el resultado es ' + str(resultado) + ' fahrenheit!')
print('[-------!]varaibles internas del modelo')
print('[-------!]',capa.get_weights())

