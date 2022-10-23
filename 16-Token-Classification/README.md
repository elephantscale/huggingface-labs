# Token classification

In this lab we will practice token classification with Transformers and HuggingFace


Lab Goals:

* Classify tokens in a sentence

### STEP 1) Prepare the data

* To load the CoNLL-2003 dataset, we use the load_dataset() method from the 🤗 Datasets library:

```python
from datasets import load_dataset
raw_datasets = load_dataset("conll2003")
```

* You will see the following output:

![](../images/20-data.png)

* Note that the data will be reused
* Verify that the data is reused:

```python
raw_datasets = load_dataset("conll2003")
```

* Look at the data

```python
raw_datasets
```


* Let’s have a look at the first element of the training set:
```python
raw_datasets["train"][0]["tokens"]
```

```text
['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.']
```

* Since we want to perform named entity recognition, we will look at the NER tags:

```python
raw_datasets["train"][0]["ner_tags"]
```

```text
[3, 0, 7, 0, 0, 0, 7, 0, 0]
```

* Those are the labels as integers ready for training, but they’re not necessarily useful when we want to inspect the data. Like for text classification, we can access the correspondence between those integers and the label names by looking at the features attribute of our dataset:

```python
ner_feature = raw_datasets["train"].features["ner_tags"]
ner_feature
```

```text
Sequence(feature=ClassLabel(num_classes=9, names=['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'], names_file=None, id=None), length=-1, id=None)
```


### STEP 2) Verify the environment