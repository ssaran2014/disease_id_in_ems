#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from flask import Flask, request, jsonify, render_template
from disease_id_in_ems.transcriber import Transcription
from disease_id_in_ems.disease_identifier import Disease

app = Flask(__name__)

@app.route("/", methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        f = request.files['audio_data']
        with open('audio.wav', 'wb') as audio:
            f.save(audio)
        print('file uploaded successfully')

        text = Transcription().indicate_transcription()
        disease_response = jsonify(Disease(text).top_disease())
        #print(type(disease_response))

        return disease_response #render_template('stoflo.html', request='POST', result=disease_response)
    else:
        return render_template('stoflo.html')


if __name__ == "__main__":
    app.run(debug=True)