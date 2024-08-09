import torch
from transformers import pipeline, set_seed

set_seed(32)
generator = pipeline(
    'text-generation', model="facebook/opt-2.7b", do_sample=True, device="cuda")
while True:
    try:
        inp = input("Q: ")
        result = generator(inp)
        print(f"A: {result}")
    except KeyboardInterrupt:
        break


# import torch
# from transformers import pipeline, set_seed, AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
# encoded_input = tokenizer(
#     "Do not meddle in the affairs of wizards, for they are subtle and quick to anger.")
# # print(encoded_input)
# # print(tokenizer.decode(encoded_input["input_ids"]))
# print([[x, tokenizer.decode(x)] for x in encoded_input['input_ids']])


# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
# s = "Do not meddle in the affairs of wizards, for they are subtle and quick to anger."
# encoded_input = tokenizer(s)
# # print(encoded_input)
# # print(tokenizer.decode(encoded_input["input_ids"]))
# print([[x, tokenizer.decode(x)] for x in encoded_input['input_ids']])


# from transformers import AutoModelForSequenceClassification

# model = AutoModelForSequenceClassification.from_pretrained(
#     "google-bert/bert-base-cased", num_labels=5)


# from datasets import load_dataset
# import numpy as np
# import evaluate
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# dataset = load_dataset("yelp_review_full")
# tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")


# def tokenize_function(examples):
#     return tokenizer(examples["text"], padding="max_length", truncation=True)


# tokenized_datasets = dataset.map(tokenize_function, batched=True)
# small_train_dataset = tokenized_datasets["train"].shuffle(
#     seed=42).select(range(1000))
# small_eval_dataset = tokenized_datasets["test"].shuffle(
#     seed=42).select(range(1000))

# model = AutoModelForSequenceClassification.from_pretrained(
#     "google-bert/bert-base-cased", num_labels=5)
# training_args = TrainingArguments(
#     output_dir="test_trainer", eval_strategy="epoch")
# metric = evaluate.load("accuracy")


# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)


# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=small_train_dataset,
#     eval_dataset=small_eval_dataset,
#     compute_metrics=compute_metrics,
# )
# trainer.train()
