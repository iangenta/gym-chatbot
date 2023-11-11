from config import MODEL_PATH, INTENTS_PATH
import nltk
import json
import random
import string
import tensorflow as tf
import spacy
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.chunk import ne_chunk
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
nlp = spacy.load("en_core_web_sm")

nltk.download('words')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')

def pad_sequence_for_input(text, tokenizer, max_length):
    # Tokenizar y preprocesar el texto con NLTK
    tokens = word_tokenize(text.lower())
    sequence = tokenizer.texts_to_sequences([tokens])
    padded_sequence = pad_sequences(sequence, maxlen=max_length)
    return np.array(padded_sequence)

# Cargar intenciones desde el archivo JSON
with open(INTENTS_PATH) as file:
    data = json.load(file)
intents = data['intents']

# Extracción de patrones y respuestas
patterns = []
responses = []
tags = []

for intent in intents:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])
    responses.append(intent['responses'])

# Tokenización y preprocesamiento del texto con NLTK
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words and word not in string.punctuation]
    return ' '.join(tokens)

patterns = [preprocess_text(pattern) for pattern in patterns]

# Crear secuencias y etiquetas
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(patterns)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(patterns)

# Padding para hacer las secuencias del mismo tamaño
padded_sequences = pad_sequences(sequences)

# Convertir las etiquetas a números enteros únicos
tag_index = {tag: idx for idx, tag in enumerate(set(tags))}
tags = [tag_index[tag] for tag in tags]

# Crear el modelo del chatbot
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(word_index) + 1, output_dim=16, input_length=padded_sequences.shape[1]),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(tag_index), activation='softmax')
])



# Compilar el modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Convertir las etiquetas a un array numpy
tags_array = np.array(tags)

# Entrenar el modelo (usando secuencias y etiquetas)
model.fit(padded_sequences, tags_array, epochs=200)
print("Etiquetas después del entrenamiento:", set(tags))
model.save('chatbot_model.h5')
print("Modelo guardado correctamente.")

# Funciones para procesar entidades y acciones específicas
def extract_entities(text):
    doc = nlp(text)
    
    entities = {}
    for ent in doc.ents:
        entities[ent.label_] = ent.text
    
    return entities
def process_schedule_workout(entities):
    if 'DATE' in entities and 'WORKOUT_TYPE' in entities:
        date = entities['DATE']
        workout_type = entities['WORKOUT_TYPE']
        response = f"Entendido. Has programado un entrenamiento de '{workout_type}' para el {date}. ¡Buena suerte!"
    else:
        response = "Lo siento, no pude entender la solicitud de programación de entrenamiento. Por favor, proporciona una fecha y un tipo de entrenamiento."
    
    return response

# Manejar la intención 'schedule_workout'
def get_bot_response(user_input):
    user_input = preprocess_text(user_input)
    sequence = tokenizer.texts_to_sequences([user_input])
    padded_sequence = pad_sequences(sequence, maxlen=padded_sequences.shape[1], padding='post')
    predicted_probs = model.predict(padded_sequence)
    predicted_label = tags[np.argmax(predicted_probs)]

    # Funciones adicionales para manejar acciones específicas
    if predicted_label == 'schedule_workout':
        entities = extract_entities(user_input)
        response = process_schedule_workout(entities)
        return response


