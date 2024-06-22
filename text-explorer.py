from flask import Flask, request, jsonify
import torch
import model_loader as mldr

app = Flask(__name__)
BookPath = "" # Initialising it here makes it global.

def LoadBook(pathToTxt):
    with open(pathToTxt, 'r', encoding='utf-8') as file:
        return file.read()

@app.route('/ask', methods=['POST'])
def AskQuestion():
    data = request.json
    context = BookPath
    question = data['question']

    # Answer with BERT.
    inputs = mldr.bertTokeniser.encode_plus(question, context, return_tensors='pt')
    inputIds = inputs['input_ids']
    attentionMask = inputs['attention_mask']
    outputs = mldr.bertModel(inputIds, attention_mask=attentionMask)
    startScores, endScores = outputs.start_logits, outputs.end_logits
    allTokens = mldr.bertTokeniser.convert_ids_to_tokens(inputIds[0].tolist())
    bertAnswer = ' '.join(allTokens[torch.argmax(startScores): torch.argmax(endScores) + 1])

    # Answer with RoBERTa.
    inputs = mldr.robertaTokeniser.encode_plus(question, context, return_tensors='pt')
    inputIds = inputs['input_ids']
    attentionMask = inputs['attention_mask']
    outputs = mldr.robertaModel(inputIds, attention_mask=attentionMask)
    startScores, endScores = outputs.start_logits, outputs.end_logits
    allTokens = mldr.robertaTokeniser.convert_ids_to_tokens(inputIds[0].tolist())
    robertaAnswer = ' '.join(allTokens[torch.argmax(startScores): torch.argmax(endScores) + 1])

    # Answer with T5.
    inputText = f"question: {question}\ncontext: {context}"
    inputs = mldr.t5Tokeniser.encode(inputText, return_tensors='pt')
    t5Answer = mldr.t5Tokeniser.decode(outputs[0], skip_special_tokens=True)

    answers = {
        "BERT": bertAnswer,
        "RoBERTa": robertaAnswer,
        "T5": t5Answer
    }
    
    return jsonify(answers)

    if __name__ == '__main__':
        BookPath = 'Parsitology_book_2.txt'
        app.run(debug=True)