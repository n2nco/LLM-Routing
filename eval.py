print('Loading transformers...')
from transformers import BertTokenizer, TFBertForSequenceClassification, pipeline

print('Loading TensorFlow...')
from tensorflow.nn import softmax

id2label = {
  0: "roleplay",
  1: "intelligent",
  2: "explicit"
}
label2id = {
  "roleplay": 0,
  "intelligent": 1,
  "explicit": 2
}

print('Loading model...')
model = TFBertForSequenceClassification.from_pretrained('./model/')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# This is to stop syntax highlighting from thinking model is a tuple
if type(model) is not TFBertForSequenceClassification:
  raise ValueError('Model is not a TFBertForSequenceClassification')

def predict(text):
  inputs = tokenizer(text, return_tensors='tf')
  logits = model(**inputs).logits

  # Softmax squishes all the output scores into a range from 0-1
  # But it also makes all the scores add up to 1
  # (Meaning we can treat them as probabilities now)
  quantized_logits = softmax(logits)

  logit_array = quantized_logits.numpy()[0]
  
  probabilities = { id2label[i]: logit_array[i] for i in range(logit_array.shape[0]) }
  return probabilities

print('Loading demo...')
print(predict("Write me code for a python server that can handle 1000 requests per second"))