from flask import Flask, render_template, request, jsonify
from chatbot import model, extract_entities, process_schedule_workout, pad_sequence_for_input, tokenizer, padded_sequences
from config import INTENTS_PATH
import random 
import json 
import numpy as np 

app = Flask(__name__)

# Cargar intenciones desde el archivo JSON
with open(INTENTS_PATH) as file:
    data = json.load(file)
intents = data['intents']

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_bot_response():
    user_input = request.form["user_input"]

    entities = extract_entities(user_input)

    predicted_probs = model.predict(pad_sequence_for_input(user_input, tokenizer, padded_sequences.shape[1]))
    predicted_label = np.argmax(predicted_probs)

    responses = intents[predicted_label]["responses"]

    if not responses:
        return jsonify({"response": "No hay respuesta disponible para esta etiqueta"})

    if "tags" in intents[predicted_label] and "schedule_workout" in intents[predicted_label]["tags"]:
        response = process_schedule_workout(entities)
        return jsonify({"response": response})

    response = random.choice(responses)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
