# Fine-Tuning

* In this lab we will fine-tune a pre-trained model.

#### Lab Goals:

* TODO
* TODO

### Step 1: Summary of the steps done before

* This will be especially useful if you are switching to Google Colab.
* 
```python
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

### Step 2: Define the training set

```python
from transformers import TrainingArguments
training_args = TrainingArguments("test-trainer")
```

### Step 3: Create a model

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
```

### Step 4: Create a trainer
    
```python   
from transformers import Trainer

trainer = Trainer(
model,
training_args,
train_dataset=tokenized_datasets["train"],
eval_dataset=tokenized_datasets["validation"],
data_collator=data_collator,
tokenizer=tokenizer,
)
```

### Step 5: Train!

```python
trainer.train()
```

