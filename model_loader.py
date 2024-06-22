from transformers import BertTokenizer, BertForQuestionAnswering
from transformers import RobertaTokenizer, RobertaForQuestionAnswering
from transformers import T5Tokenizer, T5ForQuestionAnswering

# Loading the BERT tokeniser and model.
bertTokeniser = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
bertModel = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# Loading the RoBERTa tokeniser and model.
robertaTokeniser = RobertaTokenizer.from_pretrained('deepset/roberta-base-squad2')
robertaModel = RobertaForQuestionAnswering.from_pretrained('deepset/roberta-base-squad2')

# Loading the T5 tokeniser and model.
t5Tokeniser = T5Tokenizer.from_pretrained('t5-base')
t5Model = T5ForQuestionAnswering.from_pretrained('t5-base')
