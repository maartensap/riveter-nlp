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

### Requirements 

- Python 3.8
- numpy
- pandas
- neuralcoref
- spaCy 2.3.9 (this version is required for neuralcoref)
- seaborn
- matplotlib

### Installation instructions

We strongly recommend creating a new virtual environment. Activate this environment before installing and before running the code.

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

## Authorship and Citation

This package was created by an interdiscplinary team including [Maria Antoniak](https://maria-antoniak.github.io/), [Anjalie Field](https://anjalief.github.io/), Ji Min Mun, [Melanie Walsh](https://melaniewalsh.org/), [Lauren F. Klein](https://lklein.com/), and [Maarten Sap](https://maartensap.com/).

<br>
