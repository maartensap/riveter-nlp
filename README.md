# ConnotationFramer

<br>

Package to extract connotation frames from a csv file and explore the output. 

<br>

## Installation

Requirements 
- Python 3.8
- numpy
- pandas
- spaCy 2.3.9
- neuralcoref

### Example installation instructions

We recommend creating a new virtual environment. Activate this environment before installing and before running the code.

```
conda create -n connoFramerEnv python=3.8
conda activate connoFramerEnv
```

Download this repo.

```
git clone https://github.com/maartensap/connotationFramer.git
cd connotationFramer
```

*Note: If installing on a Mac, you will need Xcode installed to run git from the command line.*

Install `neuralcoref` from source. This will also install spaCy 2.3.9.
```
git clone https://github.com/huggingface/neuralcoref.git
cd neuralcoref
pip install -r requirements.txt
pip install -e .
```

Install pandas and download spaCy files.
```
conda install pandas
python -m spacy download en_core_web_sm
```

<br>


## Usage

### conno_framer.py

To run `conno_framer.py`, see the examples in `demo.ipynb` (both located in the `conno_framer` directory).

```
framer = ConnoFramer()  
framer.load_sap_lexicon('power')
framer.train(texts,
             text_ids)
persona_score_dict = framer.get_score_totals()  
```

*Note: [Here](https://towardsdatascience.com/get-your-conda-environment-to-show-in-jupyter-notebooks-the-easy-way-17010b76e874) are some instructions for how to run `demo.ipynb` from the connoFramerEnv.*

### main.py (DEPRECATED)
To extract connotation frames from the command line, run main.py.

Command to run main.py: `python3 main.py --input_file fakeStories.csv`

Replace `fakeStories.csv` with any .csv input file, formatted as follows:

| text_id | text |
| ------- | ---- |
| 1			| This is a sample line of text. |
| 2 		| This is another line of text. It can be more than one sentence. |

*Note that each text_id should be unique, but can have any value.* 


