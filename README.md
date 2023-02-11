# ConnotationFramer

<br>

Package to extract connotation frames

<br>

## Installation

- requirements Python 3.8, numpy, pandas, spaCy 2.3.9, neuralcoref

Best to create a new virtual environment. Then download this repo

```
conda create -n py38 python=3.8
git clone git@github.com:maartensap/connotationFramer.git
cd connotationFramer
```

Install `neuralcoref` from source. This will also install spaCy 2.3.9
```
git clone https://github.com/huggingface/neuralcoref.git
cd neuralcoref
pip install -r requirements.txt
pip install -e .
```
Install pandas and download spaCy files

```
conda install pandas
python -m spacy download en_core_web_sm
```

<br>


## Usage

### main2.py
To run main2.py: see `demo.ipynb`

```
framer = ConnoFramer()  
framer.load_lexicon(lexicon_path, 'verb', 'power')
framer.train(texts,
             text_ids)
persona_score_dict = framer.get_score_totals()  
```

### main.py
Command to run main.py: `python3 main.py --input_file fakeStories.csv`

<br>

## To Do 🗒️

- Before ACL demo deadline:
   - ✅ adding agency
   - 🔲 adding other connotation frames (possibly)
   - 🔲 add methods to get e.g. all the docs in which a persona was referenced, all the docs in which a verb was used
   - 🔲 return the list of matched verbs (also easier if driven by test cases)?
   - 🔲 add in more people lists other than doctor (that's going to be easier if driven by test cases)
   - 🔲 adding in wildcard matches (related to conjugation?)
   - 🔲 allow lexicon matches for people instead of coref (e.g., if I wanted scores for all the pronouns)
   
- If time before ACL demo deadline, but probably afterwards
   - 🔲 adding other verb-related lexica (Verbnet??)
   - 🔲 add in weighted lexicon (not sure what the goal would here, but a possible extension, esp. if someone ends up using a model to create a lexicon?)
   - 🔲 allow non-coref option using persona lexicon (e.g., if I just wanted scores for all pronouns)



