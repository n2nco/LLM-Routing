print('Importing modules...')
import jsonlines
import numpy as np
print('Loading tensorflow...')
import tensorflow as tf

# Known bug where tensorflow.keras is not recognized by the syntax highlighter
# https://github.com/tensorflow/tensorflow/issues/53144
from tensorflow.keras.optimizers import Adam  # type: ignore
print('Loading transformers...')
# from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification

print('Loading training data...')
data = []
with jsonlines.open('./trainingdata/train.jsonl') as reader:
  for obj in reader:
    data.append(obj)

print('Loading tokenizer...')
tokenizer = DistilBertTokenizer.from_pretrained('bert-base-uncased')

def encode(sample):
  return tokenizer.encode_plus(
      sample['input'],
      max_length=512,
      padding='max_length',
      truncation=True,
      return_attention_mask=True,
      return_tensors='tf',
  )

print('Encoding training data...')
encoded_dataset = [encode(sample) for sample in data]

# Store dictionary keys to ensure consistent ordering
label_keys = data[0]['output'].keys(
)  # Get keys from first label, assuming all labels follow the same structure

# Extract label values in consistent order
labels = []

# Loop through all training samples
for sample in data:
  sample_labels = sample['output']
  labels.append(list(sample_labels.values()))

labels = np.array(labels)

print('Loading model...')
model = TFDistilBertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=3,  # Adjust this to match the number of classes in your output
)

# This is to stop syntax highlighting from thinking model is a tuple
if type(model) is not TFDistilBertForSequenceClassification:
  raise ValueError('Model is not a TFBertForSequenceClassification')

# Convert encoded dataset to TensorFlow tensors
inputs = [item['input_ids'] for item in encoded_dataset]
masks = [item['attention_mask'] for item in encoded_dataset]

# tf.concat combines a list of all training samples into one long tensor with all of them
inputs = tf.concat(inputs, axis=0)
masks = tf.concat(masks, axis=0)

labels = tf.convert_to_tensor(labels)

# The model will take in batch_size training examples,
# calculate the loss (error) for all of them,
# and then perform a gradient descent step
# to adjust the weights in the direction that reduces that error
# High batch sizes can cause out of memory errors
# If you have enough RAM, higher batch sizes (32 is a good number) generally lead to better results and faster learning
batch_size = 20


def create_dataset(input_ids, attention_masks, labels):
  """Create a dataset from input tensors."""
  dataset = tf.data.Dataset.from_tensor_slices(({
      "input_ids":
      input_ids,
      "attention_mask":
      attention_masks
  }, labels))
  dataset = dataset.shuffle(len(labels)).batch(batch_size)
  return dataset


train_dataset = create_dataset(inputs, masks, labels)

# Create optimizer and loss function
optimizer = Adam(learning_rate=1e-5)

# Compile the model
print('Compiling model...')
model.compile(optimizer=optimizer,
              loss="BinaryCrossentropy",
              metrics=['accuracy'])

# Save a tiny bit of RAM
# Replit kills the process when it goes above its RAM limit
# So saving RAM is important for running on Replit
del inputs, masks, labels, encoded_dataset, data

# Documentation on saving and loading models
# https://www.tensorflow.org/tutorials/keras/save_and_load

# Train the model
print('Training model...')
model.fit(train_dataset, epochs=4)

# Save the entire model, this includes everything needed to resume training
print('Exporting model...')
model.save_pretrained('./model/')
# You can load it again (in either TensorFlow or PyTorch) with
# BertForSequenceClassification.from_pretrained('./model/', from_tf=True)

# This JUST saves the weights,
# you can't resume training with it
# but it is smaller and can be used for inference
model.save_weights('model.ckpt')

# You can convert it to a SafeTensors file like this:
ckpt = tf.train.Checkpoint(model=model)
# Convert the TensorFlow weights to Safetensors format
import safetensors
safetensors.save(ckpt, 'model.safetensors')


# Here's a few advantages to storing your model as a Safetensors file instead of ckpt:
# ckpt files can contain malware, spyware, etc.
# SafeTensors can load models faster than ckpt files
# SafeTensors can be used to serialize tensors in Python, but then can be loaded in other languages and platforms, including C++, Java, and JavaScript
# Smaller file size than with ckpt files
# SafeTensors uses a checksum mechanism to ensure that serialized tensors are not corrupted