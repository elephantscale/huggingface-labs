# Transformers

* In this lab we will practice the basic use of Transformers

#### Lab Goals:

* Use Transformers for:
  * Classification
  * Text Generation
  * Entity name recognition
  * Question answering
  * Summarization
  * Translation

### How to do this lab

* Login to the server
* On the command line, open the Python interpreter
* We are not using the Jupyter notebook for this lab, and here is why:
  * Although the Jupyter notebook is a great tool, it is not the recommended practice for ML in production
  * Python scripts are better for performance
  * Architecting your systems in Python leads to better architecture and manageable code development
  * So, let us start

### Quizzes
* When you encounter a quiz, please provide your answer in private chat.
* For example
  * Quiz 1: What was your score?
  * You: `1: 99%`

### Step 1: Use a Pipeline for classification

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier("I've been waiting for a HuggingFace course my whole life.") 
```

* Your output should be similar to this:

```text
[{'label': 'POSITIVE', 'score': 0.9598047137260437}]
```

### Quiz 1: What was your score?

### Step 2: Use a Pipeline for text generation

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification")
classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)
```

* Your output should be similar to this:

```text
{'sequence': 'This is a course about the Transformers library',
 'labels': ['education', 'business', 'politics'],
 'scores': [0.8445963859558105, 0.111976258456707, 0.043427448719739914]}
```

### Quiz 2: What classification did you get?

### Step 3: Use a Pipeline for text generation

```python
from transformers import pipeline

generator = pipeline("text-generation")
generator("In this course, we will teach you how to")
Copied
[{'generated_text': 'In this course, we will teach you how to understand and use '
                    'data flow and data interchange when handling user data. We '
                    'will be working with one or more of the most commonly used '
                    'data flows — data flows of various types, as seen by the '
                    'HTTP'}]
```