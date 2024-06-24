from flask import Flask, request, jsonify
from transformers import pipeline
from LocalParser.Parser import Parser
import numpy as np

app = Flask(__name__)
myParser = Parser()

# Loading the models using pipeline.
bioRobertaQA = pipeline(task='question-answering', model='allenai/biomed_roberta_base')
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
    bioRobertaAnswer = bioRobertaQA(question=question, context=context)
    bioRobertaAnswerSerialisable = {k: makeSerialiseable(v) for k,v in bioRobertaAnswer.items()}
    # bertAnswer = bertQa(question=question, context=context)
    # robertaAnswer = robertaQa(question=question, context=context)
    # t5Answer = t5Qa(question=question, context=context)

    # Combining the answers. TODO: choose the best one with some logic.
    print("\nWe have an answer:\n")
    combinedAnswers = {
        'bioRoberta': bioRobertaAnswerSerialisable,
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
    

    
    
    
if __name__ == '__main__':
    BookPath = "Parasitology_book_2.txt"
    myParser.loadBook(BookPath)
    app.run(debug=True, port=8080)