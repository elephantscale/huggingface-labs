# Hugging Face install

### Step 1) Install Anaconda with the GUI installer

### Step 2) Install Hugging Face

```bash
pip install transformers[torch]
```

### Step 3) Verify the installation

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('I love you'))"
```

* You should see something like this

```text
[{'label': 'POSITIVE', 'score': 0.9998656511306763}]
``` 

### Step 4) Wasn't that easy?

```
Yes!
```
