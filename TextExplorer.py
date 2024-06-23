from flask import Flask, request, jsonify
from transformers import pipeline
from LocalParser.Parser import Parser

app = Flask(__name__)
myParser = Parser()

# Loading the models using pipeline.
bertQa = pipeline(task='question-answering', model='bert-large-uncased-whole-word-masking-finetuned-squad')
robertaQa = pipeline(task='question-answering', model='deepset/roberta-base-squad2')
t5Qa = pipeline(task='question-answering', model='valhalla/t5-small-qa-qg-hl')

@app.route('/ask', methods=['POST'])
def askQuestion():
    data = request.get_json()
    question = data['question']
    context = myParser.book

    # Getting answers from the modules.
    bertAnswer = bertQa(question=question, context=context)
    robertaAnswer = robertaQa(question=question, context=context)
    t5Answer = t5Qa(question=question, context=context)

    # Combining the answers. TODO: choose the best one with some logic.
    combinedAnswers = {
        'bert': bertAnswer,
        'roberta': robertaAnswer,
        't5': t5Answer
    }

    return jsonify(combinedAnswers)
    

    
    
    
if __name__ == '__main__':
    BookPath = "Parasitology_book_2.txt"
    myParser.loadBook(BookPath)
    app.run(debug=True)