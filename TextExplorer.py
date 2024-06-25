from flask import Flask, request, jsonify
from transformers import pipeline
from LocalParser.Parser import Parser
from pyngrok import ngrok
# from google.colab import userdata # Relevant only on google colab.
import numpy as np
import os
import requests
import time
import subprocess

app = Flask(__name__)
myParser = Parser()

# ngrokToken = userdata.get('ngrok_token') # Relevant only on google colab.
ngrokToken = os.getenv('NGROK_TOKEN')
ngrok.set_auth_token(ngrokToken)

# Loading the models using pipeline.
# bioRobertaQA = pipeline(task='question-answering', model='allenai/biomed_roberta_base')
vetBertQA = pipeline(task='question-answering', model='havocy28/VetBERT')
# bertQa = pipeline(task='question-answering', model='distilbert-base-uncased-distilled-squad')
# robertaQa = pipeline(task='question-answering', model='deepset/roberta-base-squad2')
# t5Qa = pipeline(task='question-answering', model='valhalla/t5-small-qa-qg-hl')

@app.route('/ask', methods=['POST'])
def askQuestion():
    data = request.get_json()
    question = data['question']
    context = myParser.book

    # Getting answers from the modules.
    print("\nAsking bio roberta your question.\n")
    # bioRobertaAnswer = bioRobertaQA(question=question, context=context)
    # bioRobertaAnswerSerialisable = {k: makeSerialiseable(v) for k,v in bioRobertaAnswer.items()}
    vetBertAnswer = vetBertQA(question=question, context=context)
    vetBertAnswerSerialisable = {k: makeSerialiseable(v) for k,v in vetBertAnswer.items()}
    # bertAnswer = bertQa(question=question, context=context)
    # robertaAnswer = robertaQa(question=question, context=context)
    # t5Answer = t5Qa(question=question, context=context)

    # Combining the answers. TODO: choose the best one with some logic.
    print("\nWe have an answer:\n")
    combinedAnswers = {
        'vetBert': vetBertAnswerSerialisable,
        # 'bioRoberta': bioRobertaAnswerSerialisable,
        # 'bert': bertAnswer,
        # 'roberta': robertaAnswer,
        # 't5': t5Answer
    }

    return jsonify(combinedAnswers)

def makeSerialiseable(obj):
    """Gets an unknown object and makes sure it can be printed in a JSON.

    Args:
        obj (any): the output of the ai model.

    Returns:
        dict|list|str|int|float|bool: an object witih a printable type.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist() # Converting a numpy array to list.
    elif isinstance(obj,(np.int64, np.int32)):
        return int(obj) # Convertinr numpy ints to standard Python int.
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj) # Convertinr numpy floats to standard Python float.
    elif isinstance(obj, (dict, list, str, int, float, bool, type(None))):
        return obj # If it's already serialisable we just return it.
    else:
        return str(obj) # A fallback in case it's a custom object.


def GetNgrokTunnels() -> list[str]:
    ngrokApiUrl = "http://localhost:4040/api/tunnels"

    response = requests.get(ngrokApiUrl)
    if response.status_code == 200:
        tunnels = response.json()['tunnels']
        # print(f"\nOpen tunnels: {tunnels}\n")
        return [tunnel['public_url'] for tunnel in tunnels]
    else:
        print(f"Failed to fetch tunnels. Status code: {response.status_code}")
        return []



if __name__ == '__main__':
    # BookPath = "text_explorer/Parasitology_book_2.txt"
    BookPath = "Parasitology_book_2.txt"
    myParser.loadBook(BookPath)
    port = 5001
    openTunnels = GetNgrokTunnels()
    if openTunnels == []:
        publicUrl = ngrok.connect(port)
    else:
        publicUrl = openTunnels[0]
    print(f"\nPublic url: {publicUrl}\n")
    app.run(debug=True, port=port)