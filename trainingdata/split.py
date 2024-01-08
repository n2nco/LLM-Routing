# Split main.jsonl into train.jsonl and test.jsonl
import jsonlines
import random

data = []

print('Reading main.jsonl')
with jsonlines.open('./trainingdata/main.jsonl') as reader:
  for obj in reader:
    data.append(obj)

print('Shuffling data...')
random.shuffle(data)

split = int(len(data) * 0.95)

print('Writing train.jsonl...')
with jsonlines.open('./trainingdata/train.jsonl', 'w') as writer:
  for obj in data[:split]:
    writer.write(obj)

print('Writing test.jsonl...')
with jsonlines.open('./trainingdata/test.jsonl', 'w') as writer:
  for obj in data[split:]:
    writer.write(obj)
