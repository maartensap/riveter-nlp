# ConnotationFramer

<br>

Package to extract connotation frames

<br>

## Installation

- best to create a new environment
- requires Python 3.8, numpy, pandas
- requires spaCy 2.1.0 `python -m pip install spacy==2.1.0`
- then can `pip install neuralcoref`
   - see https://github.com/huggingface/neuralcoref/issues/261
   - alternatively: install locally by downloading the [git repo](https://github.com/huggingface/neuralcoref) (that's what Maarten had to do on the Windows Ubuntu subsystem)



## Usage

### main2.py
To run main2.py: see `demo.ipynb`

```
framer = ConnoFramer()  
framer.train(lexicon_path, texts, text_ids)  
persona_score_dict = framer.get_score_totals()  
```


### main.py
Command to run main.py: `python3 main.py --input_file fakeStories.csv`

<br>

## To Do 🗒️

- Before ACL demo dealine:
   - 🔲 adding agency
   - 🔲 adding other connotation frames (possibly)
   - 🔲 add methods to get e.g. all the docs in which a persona was referenced, all the docs in which a verb was used
   - 🔲 return the list of matched verbs (also easier if driven by test cases)?
   - 🔲 add in more people lists other than doctor (that's going to be easier if driven by test cases)
   - 🔲 adding in wildcard matches (related to conjugation?)
- If time before ACL demo deadline, but probably afterwards
   - 🔲 adding other verb-related lexica (Verbnet??)
   - 🔲 add in weighted lexicon (not sure what the goal would here, but a possible extension, esp. if someone ends up using a model to create a lexicon?)
   - 🔲 allow non-coref option using persona lexicon (e.g., if I just wanted scores for all pronouns)

