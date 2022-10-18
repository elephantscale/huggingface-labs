# API summarizer

* In this lab we will practice summarizing text with API

#### Lab Goals:

* Use Summarizer API
* Analyze the implementation


### Step 1: Open the Summarizer API

* [Summarizer FastAPI](http://52.14.40.92/docs#)

* Click on SummarizeText
* Click on Try it out
* Substitute "text" with the following text:

```text
House Speaker Nancy Pelosi and Democratic leaders have greenlighted a plan to craft legislation that would prohibit members of Congress from trading stock, after months of resistance to a ban by Pelosi, CNBC confirmed Wednesday. At Pelosi's direction, the House Administration Committee is working on drafting the rules, and the legislation is expected to be put up for a vote this year, likely before the November midterm elections.
```

### Step 2: Look at the output
* Did you get something like the following
```text
Members of Congress could soon be banned from trading on the stock market.
```

### Step 3
* Run the same text but a different model
* Here is a list of models you can try

```text
"google/pegasus-xsum"
"google/bigbird-pegasus-large-arxiv"
"google/pegasus-cnn_dailymail"
"google/pegasus-newsroom"
"google/pegasus-pubmed"
"google/pegasus-wikihow"
"google/pegasus-reddit_tifu"
"google/pegasus-billsum"
"google/roberta2roberta_L-24_cnn_daily_mail"
"nsi319/legal-pegasus", "Legal"
```

### Quiz 3

* Run a different model and see if you get a different result

### Step 4

* Run a summarizer with a different language

### Quiz 4

* What language did you use? What was the result?