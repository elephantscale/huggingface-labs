# ML bias

* In this lab we will deal with machine learning and AI bias.

#### Lab Goals:

* Demonstrate and explain bias in machine learning models
* Experiment with ways of mitigating bias in machine learning models


### Step 1: Open the Summarizer API

```python
from transformers import pipeline

unmasker = pipeline("fill-mask", model="bert-base-uncased")
result = unmasker("This man works as a [MASK].")
print([r["token_str"] for r in result])

result = unmasker("This woman works as a [MASK].")
print([r["token_str"] for r in result])
```