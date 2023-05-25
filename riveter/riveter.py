from collections import defaultdict
from datetime import datetime
import re
import os
import pandas as pd
import pickle

import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm

import spacy
nlp = spacy.load('en_core_web_sm')

import neuralcoref
nlp.add_pipe(neuralcoref.NeuralCoref(nlp.vocab,blacklist=False),name="neuralcoref")


NER_TAGS = ["PERSON"]

PRONOUNS = ['he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'they', 'them', 'their', 'themselves']
BASEPATH = os.path.dirname(__file__)

PRONOUN_MAP = {
    "i": ["me", "my", "mine"],
    "we": ["us", "ours", "our"],
    "you": ["yours"]
}
PRONOUN_SPECIAL_CASES = {}
for p, forms in PRONOUN_MAP.items():
    for f in forms:
        PRONOUN_SPECIAL_CASES[f] = p


def default_dict_int():
        return defaultdict(int)


def default_dict_int_2():
        return defaultdict(default_dict_int)


class Riveter:

    def __init__(self, filename=None):
        self.texts = None
        self.verb_score_dict = None
        self.persona_score_dict = None
        self.id_persona_score_dict = None
        self.id_persona_count_dict = None
        self.id_nsubj_verb_count_dict = None
        self.id_dobj_verb_count_dict = None
        self.id_persona_scored_verb_dict = None # the number of scored verbs for each document and persona
        self.entity_match_count_dict = defaultdict(default_dict_int)
        self.persona_count_dict = defaultdict(int)
        self.persona_match_count_dict = defaultdict(int)
        self.people_words = None
        self.persona_polarity_verb_count_dict = defaultdict(default_dict_int_2)

        # TODO: this should go into a load() function instead
        if filename:
            with open(filename, 'rb') as f:
                my_riveter = pickle.load(f)

            for k in my_riveter.__dict__.keys():
                if k in self.__dict__.keys():
                    setattr(self, k, getattr(my_riveter, k))


    def save(self, filename='riveter.pkl'):
        with open(filename, 'wb') as file:
            # for k, v in self.__dict__.items():
            #     if isinstance(v, dict):
            #         setattr(self, k, dict(v))
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)
            print(f'Riveter successfully saved to "{filename}"')


    # def load_lexicon(self, label):
    #     if label in ['power', 'agency']:
    #         self.load_sap_lexicon(label)
    #     else:
    #         self.load_rashkin_lexicon(label)


    def load_rashkin_lexicon(self, dimension='effect'):
        """
        label can be any of [effect, state, value, writer_perspective, reader_perspective, agent_theme_perspective, theme_agent_perspective].
        Note: the persp
        """

        from IPython import embed
        lexicon_df = pd.read_csv(os.path.join(BASEPATH, 'data/rashkin-lexicon/full_frame_info.txt'), sep='\t')

        verb_score_dict = defaultdict(default_dict_int)
        for i, _row in lexicon_df.iterrows():

            _lemma  = _row['verb'].strip()

            _score_dict = {'agent': 0, 'theme': 0}

            _score_dict['agent'] += _row.get(dimension + '(a)',0) # TODO: Should the Rashkin scores be converted to [-1, 0, 1]?
            _score_dict['theme'] += _row.get(dimension + '(t)',0)

            verb_score_dict[_lemma] = _score_dict

        self.verb_score_dict = verb_score_dict


    def load_sap_lexicon(self, dimension='power'):

        label_dict = {'power_agent':  {'agent': 1, 'theme': -1},
                      'power_theme':  {'agent': -1, 'theme': 1},
                      'power_equal':  {'agent': 0, 'theme': 0},
                      'agency_pos':   {'agent': 1, 'theme': 0},
                      'agency_neg':   {'agent': -1, 'theme': 0},
                      'agency_equal': {'agent': 0, 'theme': 0}}

        lexicon_df = pd.read_csv(os.path.join(BASEPATH, 'data/sap-lexicon/agency_power.csv'))

        verb_score_dict = defaultdict(default_dict_int)
        for i, _row in lexicon_df.iterrows():
            if not pd.isnull(_row[dimension]):
                _lemma  = _row['verb'].strip()
                verb_score_dict[_lemma] = label_dict[_row[dimension]]

        self.verb_score_dict = verb_score_dict


    def load_custom_lexicon(self, lexicon_path, verb_column, agent_column, theme_column):
        """
        Allows the user to load their own lexicon.
        Expects a TSV where one column contains the verb, one column contains the agent score, 
        and one column contains the theme score. Other columns can also exist but will not be used.
        The verb must be in the same form as Rashkin, e.g. "have" "say" "take".
        The scores must be postive and negative numbers.
        """

        from IPython import embed
        lexicon_df = pd.read_csv(lexicon_path, sep='\t')

        verb_score_dict = defaultdict(default_dict_int)
        for i, _row in lexicon_df.iterrows():

            _lemma  = _row[verb_column].strip()

            _score_dict = {'agent': 0, 'theme': 0}

            _score_dict['agent'] += _row.get(agent_column, 0) 
            _score_dict['theme'] += _row.get(theme_column, 0)

            verb_score_dict[_lemma] = _score_dict

        self.verb_score_dict = verb_score_dict


    def set_people_words(self, people_words=[], load_default=False):
        if len(people_words) == 0 and load_default:
            with open(os.path.join(BASEPATH, 'data/generic_people.txt')) as f:
                self.people_words = f.read().splitlines()
        else:
            self.people_words = people_words


    def add_people_words(self, people_word):
        self.people_words.extend([people_word])


    def train(self, texts, text_ids, persona_patterns_dict=None):

        # Hacky solution to force refresh when calling train() again
        if self.texts:
            self.texts = None
            # self.verb_score_dict = None   # not this one, this is loaded on initalizing the Riveter object
            self.persona_score_dict = None
            self.id_persona_score_dict = None
            self.id_persona_count_dict = None
            self.id_nsubj_verb_count_dict = None
            self.id_dobj_verb_count_dict = None
            self.id_persona_scored_verb_dict = None # the number of scored verbs for each document and persona
            self.entity_match_count_dict = defaultdict(default_dict_int)
            self.persona_count_dict = defaultdict(int)
            self.persona_match_count_dict = defaultdict(int)
            self.people_words = None
            self.persona_polarity_verb_count_dict = defaultdict(default_dict_int_2)

        self.texts = texts
        self.text_ids = text_ids
        self.persona_score_dict, \
            self.id_persona_score_dict, \
            self.id_persona_count_dict, \
            self.id_nsubj_verb_count_dict, \
            self.id_dobj_verb_count_dict, \
            self.id_persona_scored_verb_dict = self.__score_dataset(self.texts, self.text_ids, persona_patterns_dict)


    def get_score_totals(self, frequency_threshold=0):
        return {p: s for p, s in self.persona_score_dict.items() if self.persona_match_count_dict[p] >= frequency_threshold}
    

    def plot_scores(self, number_of_scores = 10, title = "Personas by Score", frequency_threshold=0):

        # Make scores dict into dataframe
        _normalized_dict = self.get_score_totals(frequency_threshold)
        df = pd.DataFrame(_normalized_dict.items(), columns = ["persona", "score"])
        df = df.sort_values(by = "score", ascending = False)

        # If user asks for bottom x scores, e.g. -10
        if number_of_scores < 0:
            df = df[number_of_scores:]
            df = df.sort_values(by = "score", ascending = True)

        # If user asks for top x scores, e.g. 10
        else:
            df = df[:number_of_scores]

        # Make bar plot with line at 0
        graph = sns.barplot(data= df, x="persona", y="score", color = "skyblue")
        graph.axhline(0, c = "black")
        plt.xticks(rotation=45, ha='right')
        plt.title(title)
        plt.tight_layout()


    def get_scores_for_doc(self, doc_id, frequency_threshold=0):
        return {p: s/float(self.id_persona_count_dict[doc_id][p]) 
                for p, s in self.id_persona_score_dict[doc_id].items() 
                if self.persona_count_dict[p] >= frequency_threshold}


    def plot_scores_for_doc(self, doc_id, number_of_scores = 10, title = "Personas by Score", frequency_threshold=0):

    # Make scores dict into dataframe
        _normalized_dict =  self.get_scores_for_doc(doc_id, frequency_threshold=0)
        df = pd.DataFrame(_normalized_dict.items(), columns = ["persona", "score"])
        df = df.sort_values(by = "score", ascending = False)

        # If user asks for bottom x scores, e.g. -10
        if number_of_scores < 0:
            df = df[number_of_scores:]
            df = df.sort_values(by = "score", ascending = True)

        # If user asks for top x scores, eg. 10
        else:
            df = df[:number_of_scores]

        # Make bar plot with line at 0
        graph = sns.barplot(data= df, x="persona", y="score", color = "skyblue")
        graph.axhline(0, c = "black")
        plt.xticks(rotation=45, ha='right')
        plt.title(title)
        plt.tight_layout()



    def get_persona_polarity_verb_count_dict(self):
        return dict(self.persona_polarity_verb_count_dict)


    def plot_verbs_for_persona(self, persona, figsize=None, output_path=None):

        verbs_to_plot = []
        counts_to_plot = []

        persona_polarity_verb_count_dict = self.persona_polarity_verb_count_dict

        polarity_verb_count_dict = persona_polarity_verb_count_dict[persona]

        max_count = 0
        min_count = 0

        for _count, _verb in sorted(((_count, _verb) for _verb, _count in polarity_verb_count_dict['positive'].items()), reverse=True)[:10]:
            verbs_to_plot.append(_verb)
            counts_to_plot.append(_count)
            if _count > max_count:
                max_count = _count
        for _count, _verb in sorted(((_count, _verb) for _verb, _count in polarity_verb_count_dict['negative'].items()), reverse=False)[-10:]:
            verbs_to_plot.append(_verb)
            counts_to_plot.append(-_count)
            if -_count < min_count:
                min_count = -_count

        df_to_plot = pd.DataFrame({'Count': counts_to_plot},
                                index=verbs_to_plot)

        sns.set(style='ticks', font_scale=1.1)
        if not figsize:
            fig, ax = plt.subplots(1, 1)
        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        sns.heatmap(df_to_plot,
                    linewidths=1,
                    annot_kws={'size': 13},
                    cmap='PiYG',
                    # cmap=sns.diverging_palette(0, 255, sep=77, as_cmap=True),
                    center=0,
                    annot=True,
                    cbar=False,
                    vmin=min_count,
                    vmax=max_count,
                    fmt='d',
                    # fmt='.4f',
                    square=True)
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_xticks([])
        plt.yticks(rotation=0)
        plt.title(persona)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, bbox_inches='tight')


    # TODO: this would be helpful for debugging and result inspection
    # def get_docs_for_persona(self, persona):


    def get_persona_counts(self):
        return self.persona_count_dict


    def count_personas_for_doc(self, doc_id):
        return dict(self.id_persona_count_dict[doc_id])


    def count_scored_verbs_for_doc(self, doc_id):
        return dict(self.id_persona_scored_verb_dict[doc_id])


    def count_nsubj_for_doc(self, doc_id, matched_only=False):
        """Returns the set of persona-verb pairs (where persona is the subject of the verb)
        matched_only: only returns the verbs that are in the lexicon
        """
        counts = dict(self.id_nsubj_verb_count_dict[doc_id])
        if matched_only:
            counts = {pair: cnt for pair,cnt in counts.items() if pair[1] in self.verb_score_dict}
        return counts


    def count_dobj_for_doc(self, doc_id,matched_only=False):
        """Returns the set of persona-verb pairs (where persona is the object of the verb)
        matched_only: only returns the verbs that are in the lexicon
        """
        counts = dict(self.id_dobj_verb_count_dict[doc_id])
        if matched_only:
            counts = {pair: cnt for pair,cnt in counts.items() if pair[1] in self.verb_score_dict}
        return counts


    def get_persona_cluster(self, persona):
        return dict(self.entity_match_count_dict[persona.lower()]) # TODO: possibly shouldn't lower case here?
                                                                   #       is it possible to have multiple entities whose only difference is capitalization?


    # def __loadFile(self, input_file, text_column, id_column):
    #     """@todo: make this read in multiple types of files"""
    #     df = pd.read_csv(input_file)
    #     return df[text_column].tolist(), df[id_column].tolist()


    def __getCorefClusters(self, spacyDoc):
        clusters = spacyDoc._.coref_clusters
        return clusters


    def __isSpanPerson(self,span,peopleWords):
        isPerson = len(span.ents) > 0
        isPerson = isPerson and any([e.label_ in NER_TAGS for e in span.ents])
        isPerson = isPerson or any([w.text.lower()==p.lower() for w in span for p in peopleWords])
        return isPerson


    def __isClusterPerson(self,cluster,peopleWords):
        areMentionsPeople = [self.__isSpanPerson(m,peopleWords) for m in cluster.mentions]

        if all(areMentionsPeople):
            return True

        pctMentionsPeople = sum(areMentionsPeople) / len(areMentionsPeople)
        return pctMentionsPeople >=.5


    def __getPeopleClusters(self, spacyDoc, peopleWords):

        clusters = self.__getCorefClusters(spacyDoc)

        # need to add singleton clusters for tokens detected as people
        singletons = {}

        peopleClusters = set()
        # adding I / you clusters to people
        main2cluster = {c.main.text: c for c in clusters}

        if "I" in main2cluster:
            peopleClusters.add(main2cluster["I"])
        if "you" in main2cluster:
            peopleClusters.add(main2cluster["you"])

        # This checks if each coref cluster contains a "person", and only keeps clusters that contain at least 1 person
        # it also adds singletons
        for span in spacyDoc.noun_chunks:
            isPerson = self.__isSpanPerson(span,peopleWords)
            # isPerson2 = len(span.ents) > 0 and any([e.label_ in NER_TAGS for e in span.ents])
            # isPerson2 = isPerson2 or any([w.text.lower()==p.lower() for w in span for p in peopleWords])

            if isPerson:

                # check if it's in the clusters to add people
                inClusterAlready = False
                for c in clusters:
                    if any([m.start == span.start and m.end == span.end for m in c.mentions]):
                        #print("Yes", c)
                        peopleClusters.add(c)
                        inClusterAlready = True

                # also add singletons, which may be real singletons or mis-matched ones
                if not inClusterAlready:
                    # simple heuristic to fix coref bug:
                    # if the span text is the exact same as the main cluster mention, merge

                    matched = [c for c in peopleClusters if c.main.text == span.text]
                    if len(matched) > 0:
                        matched = matched[0]
                        matched.mentions.append(span)
                    else:
                        # else, make new cluster
                        peopleClusters.add(neuralcoref.neuralcoref.Cluster(len(clusters),span,[span]))


        # Re-iterating over noun chunks, that's the entities that are going to have verbs,
        # and removing the coref mentions that are not a noun chunk
        # Note that we keep coref mentions that noun chunks but not people (as long as something else in the chain is a person)
        newClusters = {c.main:[] for c in peopleClusters}
        for span in spacyDoc.noun_chunks:
            for c in peopleClusters:
                for m in c.mentions:
                    if m.start==span.start and m.end == span.end and span.text == m.text:
                        newClusters[c.main].append(span)

        newPeopleClusters = [neuralcoref.neuralcoref.Cluster(i,main,mentions)
                            for i,(main, mentions) in enumerate(newClusters.items())]
        newPeopleClusters = [c for c in newPeopleClusters if self.__isClusterPerson(c,peopleWords)]

        return newPeopleClusters


    def __parse_and_extract_coref(self, text):
        # todo possibly: keep track of each spacy.Token / Span and return those along with the str representation, so
        # that we can use that info later to visualize the lexicon output

        nsubj_verb_count_dict = defaultdict(int)
        dobj_verb_count_dict = defaultdict(int)

        doc = nlp(text)

        if self.people_words is None:
            self.set_people_words(load_default=True)

        clusters = self.__getPeopleClusters(doc, peopleWords=self.people_words)

        for _cluster in clusters:

            _num_non_pronouns = len([_span for _span in _cluster if str(_span).lower() not in PRONOUNS])

            # If the cluster is only third person pronouns, don't include it (coref failed to match to an entity)
            # Second condition handles weird edge case where the cluster key is in PRONOUNS but the spands include "you" or "I"
            if _num_non_pronouns > 0 and str(_cluster.main).lower() not in PRONOUNS:

                for _span in _cluster:

                    _cluster_text = str(_cluster.main).lower()
                    _cluster_text = PRONOUN_SPECIAL_CASES.get(_cluster_text, _cluster_text)

                    self.persona_count_dict[_cluster_text] += 1
                    self.entity_match_count_dict[_cluster_text][str(_span).lower()] += 1

                    if _span.root.dep_ == 'nsubj':
                        _verb = _span.root.head.lemma_.lower()
                        nsubj_verb_count_dict[(_cluster_text, _verb)] += 1

                    elif _span.root.dep_ == 'dobj':
                        _verb = _span.root.head.lemma_.lower()
                        dobj_verb_count_dict[(_cluster_text, _verb)] += 1

        return nsubj_verb_count_dict, dobj_verb_count_dict


    def __parse_and_extract(self, text, persona_patterns_dict):

        nsubj_verb_count_dict = defaultdict(int)
        dobj_verb_count_dict = defaultdict(int)

        doc = nlp(text)

        for _parsed_sentence in doc.sents:
            for _noun_chunk in _parsed_sentence.noun_chunks:

                if _noun_chunk.root.dep_ == 'nsubj':

                    for _persona, _pattern in persona_patterns_dict.items():

                        if re.findall(_pattern, _noun_chunk.text.lower()):

                            self.persona_count_dict[_persona] += 1
                            self.entity_match_count_dict[_persona][_noun_chunk.text.lower()] += 1

                            _nusbj = _persona
                            _verb = _noun_chunk.root.head.lemma_.lower()
                            nsubj_verb_count_dict[(_nusbj, _verb)] += 1

                elif _noun_chunk.root.dep_ == 'dobj':

                    for _persona, _pattern in persona_patterns_dict.items():

                        if re.findall(_pattern, _noun_chunk.text.lower()):

                            self.persona_count_dict[_persona] += 1
                            self.entity_match_count_dict[_persona][_noun_chunk.text.lower()] += 1

                            _dobj = _persona
                            _verb = _noun_chunk.root.head.lemma_.lower()
                            dobj_verb_count_dict[(_dobj, _verb)] += 1

        return nsubj_verb_count_dict, dobj_verb_count_dict


    def __score_document(self,
                         nsubj_verb_count_dict,
                         dobj_verb_count_dict):

        persona_score_dict = defaultdict(float)
        persona_scored_verbs_dict = defaultdict(int)

        for (_persona, _verb), _count in nsubj_verb_count_dict.items():
            if _verb in self.verb_score_dict:
                persona_scored_verbs_dict[_persona] += 1
                _agent_score = self.verb_score_dict[_verb]['agent']
                persona_score_dict[_persona] += (_count*_agent_score)
                self.persona_match_count_dict[_persona] += 1
                if _agent_score < 0:
                    self.persona_polarity_verb_count_dict[_persona]['negative'][_verb + '_nsubj'] += 1
                elif _agent_score > 0:
                    self.persona_polarity_verb_count_dict[_persona]['positive'][_verb + '_nsubj'] += 1

        for (_persona, _verb), _count in dobj_verb_count_dict.items():
            if _verb in self.verb_score_dict:
                persona_scored_verbs_dict[_persona] += 1
                _theme_score = self.verb_score_dict[_verb]['theme']
                persona_score_dict[_persona] += (_count*_theme_score)
                self.persona_match_count_dict[_persona] += 1
                if _theme_score < 0:
                    self.persona_polarity_verb_count_dict[_persona]['negative'][_verb + '_dobj'] += 1
                elif _theme_score > 0:
                    self.persona_polarity_verb_count_dict[_persona]['positive'][_verb + '_dobj'] += 1

        return persona_score_dict, persona_scored_verbs_dict


    def __score_dataset(self, texts, text_ids, persona_patterns_dict):

        id_nsubj_verb_count_dict = {}
        id_dobj_verb_count_dict = {}
        id_persona_score_dict = {}
        id_persona_count_dict = {}
        id_persona_scored_verb_dict = {}

        for _text, _id in tqdm(zip(texts, text_ids), total=len(texts)):

            if not persona_patterns_dict:
                _nsubj_verb_count_dict, _dobj_verb_count_dict = self.__parse_and_extract_coref(_text)
            else:
                _nsubj_verb_count_dict, _dobj_verb_count_dict = self.__parse_and_extract(_text, persona_patterns_dict)

            _persona_score_dict, _persona_scored_verb_dict = self.__score_document(_nsubj_verb_count_dict, _dobj_verb_count_dict)
            _persona_count_dict = self.__get_persona_counts_per_document(_nsubj_verb_count_dict, _dobj_verb_count_dict)

            id_persona_score_dict[_id] = _persona_score_dict
            id_persona_count_dict[_id] = _persona_count_dict
            id_nsubj_verb_count_dict[_id] = _nsubj_verb_count_dict
            id_dobj_verb_count_dict[_id] = _dobj_verb_count_dict
            id_persona_scored_verb_dict[_id] = _persona_scored_verb_dict

        persona_score_dict = defaultdict(float)
        for _id, _persona_score_dict in id_persona_score_dict.items():
            for _persona, _score in _persona_score_dict.items():
                persona_score_dict[_persona] += _score

        # Normalize the scores over the total number of nsubj and dobj occurrences in the dataset for this persona
        persona_score_dict = {p: s/float(self.persona_count_dict[p]) for p, s in persona_score_dict.items() if self.persona_count_dict[p] > 0}

        print(str(datetime.now())[:-7] + ' Complete!')
        return persona_score_dict, id_persona_score_dict, id_persona_count_dict, id_nsubj_verb_count_dict, id_dobj_verb_count_dict, id_persona_scored_verb_dict


    def __get_persona_counts_per_document(self,
                                          nsubj_verb_count_dict,
                                          dobj_verb_count_dict):

        persona_count_dict = defaultdict(int)

        for (_persona, _verb), _count in nsubj_verb_count_dict.items():
            persona_count_dict[_persona] += _count
        for (_persona, _verb), _count in dobj_verb_count_dict.items():
            persona_count_dict[_persona] += _count

        return persona_count_dict


    def __evaluate_verb_coverage(self, id_nsubj_verb_count_dict):

        verb_count_dict = defaultdict(int)

        for _id, _nsubj_verb_count_dict in id_nsubj_verb_count_dict.items():
            for (_persona, _verb), _count in _nsubj_verb_count_dict.items():
                verb_count_dict[_verb] += 1

        return verb_count_dict
