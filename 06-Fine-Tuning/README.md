# Fine-Tuning

* In this lab we will fine-tune a pre-trained model.

#### Lab Goals:

* TODO
* TODO

### Step 1: Create an account on HuggingFace

* TODO
* 
### Step 2: Create a new notebook

```python
import torch
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification

# Same as before
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
]
batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

# This is new
batch["labels"] = torch.tensor([1, 1])

optimizer = AdamW(model.parameters())
loss = model(**batch).loss
loss.backward()
optimizer.step()
```

### Step 3: Load a dataset from HuggingFace

```python
from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
raw_datasets
```

* You should see something like this:

```text
DatasetDict({
    train: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 408
    })
    test: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 1725
    })
})  
```

### Step 4: Look at one phrase in the dataset

```python
raw_train_dataset = raw_datasets["train"]
raw_train_dataset[0]
```

* Your output
* (I recommend making your window wide enough to see the whole output)

![](../images/02-output.png)

### Step 5: Dislay data labels

* To know which integer corresponds to which label, we can inspect the features of our raw_train_dataset. This will tell us the type of each column:

```python
raw_train_dataset.features
```

* Your output

```text
{'sentence1': Value(dtype='string', id=None),
 'sentence2': Value(dtype='string', id=None),
 'label': ClassLabel(num_classes=2, names=['not_equivalent', 'equivalent'], names_file=None, id=None),
 'idx': Value(dtype='int32', id=None)}  
```

