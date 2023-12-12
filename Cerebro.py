import random
import time
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation 
from tensorflow.keras.optimizers import RMSprop

# Obtener la ruta del directorio actual donde se encuentra main.py
current_directory = os.path.dirname(os.path.abspath(__file__))

# Construir la ruta al archivo MODEL.TXT
filepath = os.path.join(current_directory, 'MODEL.TXT')

#Leer el archivo y definir desde el inicio hasta el final
texto = open(filepath, 'rb').read().decode(encoding='ISO-8859-1').lower()
texto = texto[1:100000]

letras = sorted(set(texto))
texto_a_indice = dict((c, i) for i, c in enumerate(letras))
indice_a_letra = dict((i, c) for i, c in enumerate(letras))

tamaño_sec = 40
paso_tamaño = 3

oraciones = []
siguiente_letra = []

# Aquí se corrige cómo se generan las oraciones y sus siguientes letras
for i in range(0, len(texto) - tsamaño_sec, paso_tamaño):
    oraciones.append(texto[i: i + tamaño_sec])
    siguiente_letra.append(texto[i + tamaño_sec])  # Solo la siguiente letra

# Corrección en la inicialización de x e y usando np.bool_ o bool
x = np.zeros((len(oraciones), tamaño_sec, len(letras)), dtype=np.bool_)
y = np.zeros((len(oraciones), len(letras)), dtype=np.bool_)

# Rellenando las matrices x e y
for i, oracion in enumerate(oraciones):
    for t, letra in enumerate(oracion):
        x[i, t, texto_a_indice[letra]] = 1
    y[i, texto_a_indice[siguiente_letra[i]]] = 1  # Aquí también se corrige

# Continuación del código para construir y entrenar el modelo...
model = Sequential()
# ...

model.add(LSTM(256, input_shape=(tamaño_sec, len(letras))))
model.add(Dense(len(letras)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01))
model.fit(x,y,batch_size=256, epochs=8)

model.save('textgenerator.model')

'''
model = tf.keras.models.load_model('textgenerator.model')

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generar_text(length, temperature):
    start_index = random.randint(0, len(texto) - tamaño_sec - 1)
    generated = ''
    sentence = texto[start_index: start_index + tamaño_sec]
    generated += sentence

    for i in range(length):
        x = np.zeros((1, tamaño_sec, len(letras)), dtype=np.bool_)

        for t, char in enumerate(sentence):
            x[0, t, texto_a_indice[char]] = 1

        prediction = model.predict(x, verbose=0)[0]
        next_index = sample(prediction, temperature)
        next_character = indice_a_letra[next_index]

        generated += next_character
        sentence = sentence[1:] + next_character  # Actualiza la 'sentence'

        print(next_character, end='', flush=True)  # Imprime la letra generada sin salto de línea
        time.sleep(0.1)  # Introduce un retraso de 0.1 segundos entre letras

    return generated

# Imprimir texto generado con diferentes temperaturas

print(generar_text(1000, 0.6))
'''