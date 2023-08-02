# Riveter 💪

<br>

Riveter 💪 is a Python package that measures social dynamics between personas mentioned in a collection of texts.

The package identifies and extracts the subjects, verbs, and direct objects in texts; it performs coreference resolution on the personas mentioned in the texts (e.g., clustering "Elizabeth Bennet" and "she" together as one persona); and it measures social dynamics between the personas by referencing a given lexicon. The package currently includes lexica for Maarten Sap et al's ***connotation frames of power and agency*** and Rashkin et al's ***connotation frames of perspective, effect, value, and mental state***, but you can also load your own custom lexicon.

The name Riveter is inspired by ["Rosie the Riveter,"](https://en.wikipedia.org/wiki/File:We_Can_Do_It!.jpg) the allegorical figure who came to represent American women working in factories and at other industrial jobs during World War II. Rosie the Riveter has become an iconic symbol of power and shifting gender roles — subjects that the Riveter package aims to help users measure and explore.    

<br>

## Demo video and notebook

Watch our two minute demo video here: [link](https://youtu.be/Uftyd8eCmFw)

Check out our demo notebook here: [link]([https://github.com/maartensap/riveter-nlp/blob/main/riveter/demo.ipynb](https://colab.research.google.com/drive/19akZ2Qu7uva8jOsc49e_2HJmDo88WAXm?usp=sharing))

<br>

## Installation

### Quick start

To skip local installation and get started immediately, you can using [this Google Colab notebook](https://colab.research.google.com/drive/19akZ2Qu7uva8jOsc49e_2HJmDo88WAXm?usp=sharing).

### Requirements 

- Python 3.9
- numpy
- pandas
- seaborn
- matplotlib
- spacy-experimental

### Installation instructions

These instructions have been tested on OSX machines. We have not tested these instructions in other environments.

1. We strongly recommend creating a new virtual environment. Activate this environment before installing and before running the code.

```bash
conda create -n riveterEnv python=3.9
conda activate riveterEnv
```

2. Download this repo by using the Git command below or by downloading the repository manually (click the green _Code_ button above, select _Download ZIP_, and then unzip the downloaded directory). 

```bash
git clone https://github.com/maartensap/riveter-nlp.git
cd riveter-nlp
```

*Note: If installing on a Mac, you will need Xcode installed to run git from the command line.*

3. Install spacy-experimental and the spaCy model files.
```bash
pip install -U spacy-experimental
pip install https://github.com/explosion/spacy-experimental/releases/download/v0.6.0/en_coreference_web_trf-3.4.0a0-py3-none-any.whl#egg=en_coreference_web_trf
python -m spacy download en_core_web_sm
```

4. Install pandas and seaborn.
```bash
conda install pandas
conda install seaborn
```

<br>


## Usage

To use Riveter 💪, see the examples in [our demo notebook](https://github.com/maartensap/riveter-nlp/blob/main/riveter/demo.ipynb). 

This notebook includes both toy and realistic examples and all of the most important function calls.

If you want a quick start:
```python
riveter = Riveter()  
riveter.load_sap_lexicon('power')
riveter.train(texts,
             text_ids)
persona_score_dict = riveter.get_score_totals()  
```

*Note: [Here](https://towardsdatascience.com/get-your-conda-environment-to-show-in-jupyter-notebooks-the-easy-way-17010b76e874) are some instructions for how to run `demo.ipynb` from the riveterEnv.*

<br>

## Documentation

        
#### `get_score_totals(frequency_threshold=0)`

Get the final scores for all the entities, above some frequency threshold across the dataset.

| Name               | Type              | Description                      |
| ------------------ | ----------------- | -------------------------------- |
| `frequency_threshold` | integer | Optional: Entities must be matched to at least this many verbs to appear in the output. |
| RETURNS | dictionary | Dictionary of entities and their total scores. |

<br>

#### `plot_scores(number_of_scores=10, title="Personas by Score", frequency_threshold=0)`

Create a bar plot showing the final scores across the dataset.

| Name               | Type              | Description                      |
| ------------------ | ----------------- | -------------------------------- |
| `number_of_scores` | integer | Optional: Show only the top or bottom number of scores. |
| `title` | string | Optional: Plot title. |
| `frequency_threshold` | integer | Optional: Entities must be matched to at least this many verbs to appear in the output. |

<br>

#### `get_scores_for_doc(doc_id, frequency_threshold=0)`

Get the final scores for all the entities, above some frequency threshold in a single document.

| Name               | Type              | Description                      |
| ------------------ | ----------------- | -------------------------------- |
| `doc_id` | string or integer | Show results for this document ID. |
| `frequency_threshold` | integer | Optional: Entities must be matched to at least this many verbs to appear in the output. |
| RETURNS | dictionary | Nested dictionary of document IDs, entities, and their total scores. |

<br>

#### `plot_scores_for_doc(doc_id, number_of_scores=10, title="Personas by Score", frequency_threshold=0)`

Create a bar plot showing the final scores for a single document.

| Name               | Type              | Description                      |
| ------------------ | ----------------- | -------------------------------- |
| `doc_id` | string or integer | Show results for this document ID. |
| `number_of_scores` | integer | Optional: Show only the top or bottom number of scores. |
| `title` | string | Optional: Plot title. |
| `frequency_threshold` | integer | Optional: Entities must be matched to at least this many verbs to appear in the output. |

<br>

#### `get_persona_polarity_verb_count_dict()`

Gets all the verbs, their frequencies, and whether they contributed positively or negatively to the final scores for every entity. Computed across the whole dataset.

| Name               | Type              | Description                      |
| ------------------ | ----------------- | -------------------------------- |
| RETURNS | dictionary | Nested dictionary of entities, positive or negative contribution, verbs, and counts. |

<br>

#### `plot_verbs_for_persona(persona, figsize=None, output_path=None)`

Create a heatmap showing the verb counts for a single persona.

| Name               | Type              | Description                      |
| ------------------ | ----------------- | -------------------------------- |
| `persona` | string | The entity whose results will be shown in the plot. |
| `figsize` | tuple | Optional: Figure dimensions, e.g. (2, 4). |
| `output_path` | string | Optional: Where to save the plot as a file. |

<br>

#### `get_persona_counts()`

Get the total counts for the entities (all entity matches, whether or not they were matched to a lexicon verb).

| Name               | Type              | Description                      |
| ------------------ | ----------------- | -------------------------------- |
| RETURNS | dictionary | Dictionary of entities and integer counts. |

<br>

#### `count_personas_for_doc(doc_id)`

Get the entity counts for a single document.

| Name               | Type              | Description                      |
| ------------------ | ----------------- | -------------------------------- |
| doc_id | string or integer | Show results for this document ID |
| RETURNS | dictionary | Dictionary of entities and integer counts. |

<br>

#### `count_scored_verbs_for_doc(doc_id)`

Get the verb counts (verbs that were matched to the lexicon) for a single document.

| Name               | Type              | Description                      |
| ------------------ | ----------------- | -------------------------------- |
| doc_id | string or integer | Show results for this document ID |
| RETURNS | dictionary | Dictionary of verbs and integer counts. |

<br>

#### `count_nsubj_for_doc(doc_id, matched_only=False)`

Get the noun subject counts for a single document.

| Name               | Type              | Description                      |
| ------------------ | ----------------- | -------------------------------- |
| doc_id | string or integer | Show results for this document ID |
| matched_only | boolean | If true, return only the subjects that were matched to identified entities. |
| RETURNS | dictionary | Dictionary of noun subjects and integer counts. |

<br>

#### `count_dobj_for_doc(doc_id,matched_only=False)`

Get the direct object counts for a single document.

| Name               | Type              | Description                      |
| ------------------ | ----------------- | -------------------------------- |
| doc_id | string or integer | Show results for this document ID |
| matched_only | boolean | If true, return only the direct objects that were matched to identified entities. |
| RETURNS | dictionary | Dictionary of direct object and integer counts. |

<br>

#### `get_persona_cluster(persona)`

Get the full entity cluster from `neuralcoref`.

| Name               | Type              | Description                      |
| ------------------ | ----------------- | -------------------------------- |
| persona | string | Show results for this entity. |
| RETURNS | dictionary | Dictionary of the main entity string and all of its string matches. |

<br>

#### `load_sap_lexicon(dimension='power')`

Load the verb lexicon from Sap et al., 2017.

| Name               | Type              | Description                      |
| ------------------ | ----------------- | -------------------------------- |
| dimension | string | Select the lexicon: "power" or "agency". |

<br>

#### `load_rashkin_lexicon(dimension='effect')`

Load the verb lexicon from Rashkin et al., 2016.

| Name               | Type              | Description                      |
| ------------------ | ----------------- | -------------------------------- |
| dimension | string | Select the lexicon: ["effect", "state", "value", "writer_perspective", "reader_perspective", "agent_theme_perspective", "theme_agent_perspective"]. |

<br>

#### `load_custom_lexicon(lexicon_path, verb_column, agent_column, theme_column)`

Load your own verb lexicon.

| Name               | Type              | Description                      |
| ------------------ | ----------------- | -------------------------------- |
| lexicon_path | string | Path the lexicon; this should be a TSV file. |
| verb_column | string | Column in the TSV that contains the verb. This should be in the same form as the Rashkin lexicon, e.g. "have" "take". |
| agent_column | string | Column containing the agent score (positive or negative number). |
| theme_column | string | Column containing the theme score (positive or negative number). |

<br>

#### `get_documents_for_verb(target_verb)`

Find all the documents matched to the verb.

| Name               | Type              | Description                      |
| ------------------ | ----------------- | -------------------------------- |
| target_verb | string | The verb you'd like to match. |
| RETURNS | (list, list) | List of matched document IDs, list of matched document texts. |

<br>

#### `get_documents_for_persona(target_persona)`

Find all the documents matched to the persona.

| Name               | Type              | Description                      |
| ------------------ | ----------------- | -------------------------------- |
| target_persona | string | The persona you'd like to match. |
| RETURNS | (list, list) | List of matched document IDs, list of matched document texts. |

<br>

## Authorship and Citation

This package was created by an interdiscplinary team including [Maria Antoniak](https://maria-antoniak.github.io/), [Anjalie Field](https://anjalief.github.io/), Jimin Mun, [Melanie Walsh](https://melaniewalsh.org/), [Lauren F. Klein](https://lklein.com/), and [Maarten Sap](https://maartensap.com/). You can find our paper writeup at the following URL: http://maartensap.com/pdfs/antoniak2023riveter.pdf

Use the following BibTex to cite the paper:
```bibtex
@article{antoniak2023riveter,
  title={Riveter: Measuring Power and Social Dynamics Between Entities},
  author={Antoniak, Maria and Field, Anjalie and Mun, Ji Min and Walsh, Melanie and Klein, Lauren F. and Sap, Maarten},
  year={2023},
  url={http://maartensap.com/pdfs/antoniak2023riveter.pdf}
}
```

<br>
