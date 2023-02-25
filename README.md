# Riveter 💪

<br>

The Riveter 💪 package measures social dynamics between personas mentioned in a collection of texts.

The package identifies and extracts the subjects, verbs, and direct objects in texts; it performs coreference resolution on the personas mentioned in the texts (e.g., clustering "Elizabeth Bennet", "Lizzy," and "she" together as one persona); and it measures social dynamics between the personas by referencing a given lexicon. The package currently includes Maarten Sap et al's lexicon for power and agency and Rashkin et al's lexicon for perspective, effect, value, and mental state. 

The name Riveter is inspired by ["Rosie the Riveter,"](https://en.wikipedia.org/wiki/File:We_Can_Do_It!.jpg) the allegorical figure who came to represent American women working in factories and at other industrial jobs during World War II. Rosie the Riveter has become an iconic symbol of power and shifting gender roles — subjects that the Riveter package aims to help users measure and explore.    

<br>

## Demo video

Watch our two minute demo video here: [https://youtu.be/Uftyd8eCmFw](https://youtu.be/Uftyd8eCmFw)

<br>

## Installation

Requirements 
- Python 3.8
- numpy
- pandas
- spaCy 2.3.9
- neuralcoref
- seaborn
- matplotlib

### Example installation instructions

We recommend creating a new virtual environment. Activate this environment before installing and before running the code.

```bash
conda create -n riveterEnv python=3.8
conda activate riveterEnv
```

Download this repo.

```bash
git clone https://github.com/maartensap/riveter-nlp.git
cd riveter-nlp
```

*Note: If installing on a Mac, you will need Xcode installed to run git from the command line.*

Install `neuralcoref` from source. This will also install spaCy 2.3.9.
```bash
git clone https://github.com/huggingface/neuralcoref.git
cd neuralcoref
pip install -r requirements.txt
pip install -e .
cd ..
```

Install pandas and download spaCy files.
```bash
conda install pandas
python -m spacy download en_core_web_sm
```

<br>


## Usage

### riveter.py

To run `riveter.py`, see the example in `demo.ipynb` (located in the `riveter` directory).

```python
riveter = Riveter()  
riveter.load_sap_lexicon('power')
riveter.train(texts,
             text_ids)
persona_score_dict = riveter.get_score_totals()  
```

*Note: [Here](https://towardsdatascience.com/get-your-conda-environment-to-show-in-jupyter-notebooks-the-easy-way-17010b76e874) are some instructions for how to run `demo.ipynb` from the riveterEnv.*

