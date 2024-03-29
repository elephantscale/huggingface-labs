# Hugging Face install

### Step 1) Install Anaconda with the GUI installer

* In Ubuntu, you can simply do this
```shell
wget https://elephantscale-public.s3.amazonaws.com/downloads/Anaconda3-2023.03-Linux-x86_64.sh
chmod +x Anaconda3-2023.03-Linux-x86_64.sh
./Anaconda3-2023.03-Linux-x86_64.sh
```
* Last install step: say **"yes"** to initializing Conda


* *(Optional, if above does not work)*
* Download the Anaconda installer for your operating system from [here](https://www.anaconda.com/products/individual#Downloads). We recommend using the graphical installer.
* Last install step: say **"yes"** to initializing Conda
* Start a new terminal for the changes to take effect

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
