# ConnotationFramer
Package to extract connotation frames


## Installation

- best to create a new environment
- requires Python 3.8, numpy, pandas
- requires spaCy 2.1.0 `python -m pip install spacy==2.1.0`
- then can `pip install neuralcoref`
- (see https://github.com/huggingface/neuralcoref/issues/261)

(alternatively: Note, this requires [NeuralCoref](https://github.com/huggingface/neuralcoref), which I had to install by downloading the git repo)


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


## To Do

- adding agency
- return the list of matched verbs (also easier if driven by test cases)?
- adding in wildcard matches (related to conjugation?)
- add in more people lists other than doctor (that's going to be easier if driven by test cases)
- adding other connotation frames (possibly)
- add in weighted lexicon (not sure what the goal would here, but a possible extension, esp. if someone ends up using a model to create a lexicon?)
- add methods to get e.g. all the docs in which a persona was referenced, all the docs in which a verb was used
