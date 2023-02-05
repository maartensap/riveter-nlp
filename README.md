# ConnotationFramer
Package to extract connotation frames


Command to run the code rn: `python3 main.py --input_file fakeStories.csv`

Note, this requires [NeuralCoref](https://github.com/huggingface/neuralcoref), which I had to install by downloading the git repo

Installation Notes:  
- best to create a new environment
- requires Python 3.8, numpy, pandas
- requires spaCy 2.1.0 `python -m pip install spacy==2.1.0`
- then can `pip install neuralcoref`
- (see https://github.com/huggingface/neuralcoref/issues/261)
