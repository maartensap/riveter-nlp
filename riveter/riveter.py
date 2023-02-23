from collections import defaultdict
from datetime import datetime

import pandas as pd

from tqdm import tqdm

import spacy
nlp = spacy.load('en_core_web_sm')

import neuralcoref
nlp.add_pipe(neuralcoref.NeuralCoref(nlp.vocab,blacklist=False),name="neuralcoref")


NER_TAGS = ["PERSON"]

PRONOUNS = ['he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'they', 'them', 'their', 'themselves']


# This is messy, but I think special-casing pronouns is probably the right thing to do
pronoun_map = {
    "i": ["me", "my", "mine"],
    "we": ["us", "ours", "our"],
    "you": ["yours"]
}
pronoun_special_cases = {}
for p, forms in pronoun_map.items():
    for f in forms:
        pronoun_special_cases[f] = p


class Riveter:

    def __init__(self):
        self.verb_score_dict = None
        self.persona_score_dict = None
        self.id_persona_score_dict = None
        self.id_persona_count_dict = None
        self.id_nsubj_verb_count_dict = None
        self.id_dobj_verb_count_dict = None
        self.id_persona_scored_verb_dict = None
        self.entity_match_count_dict = defaultdict(lambda: defaultdict(int))
        self.persona_count_dict = defaultdict(int)


    def load_lexicon(self, label):
        if label in ['power', 'agency']:
            self.load_sap_lexicon(label)
        else:
            self.load_rashkin_lexicon(label)


    def load_rashkin_lexicon(self, label='effect'):
        """
        label can be any of [effect, state, value, perspective]
        """

        lexicon_df = pd.read_csv('../data/rashkin-lexicon/full_frame_info.txt', sep='\t')

        verb_score_dict = defaultdict(lambda: defaultdict(int))
        for i, _row in lexicon_df.iterrows():

            _lemma  = _row['verb'].strip()

            _score_dict = {'agent': 0, 'theme': 0}

            _score_dict['agent'] += _row[label + '(s)'] # TODO: Should the Rashkin scores be converted to [-1, 0, 1]?
            _score_dict['theme'] += _row[label + '(o)']

            verb_score_dict[_lemma] = _score_dict

        self.verb_score_dict = verb_score_dict


    def load_sap_lexicon(self, label_column):

        label_dict = {'power_agent':  {'agent': 1, 'theme': -1}, 
                      'power_theme':  {'agent': -1, 'theme': 1},
                      'power_equal':  {'agent': 0, 'theme': 0},
                      'agency_pos':   {'agent': 1, 'theme': 0},
                      'agency_neg':   {'agent': -1, 'theme': 0},
                      'agency_equal': {'agent': 0, 'theme': 0}}

        lexicon_df = pd.read_csv('../data/sap-lexicon/agency_power.csv')

        verb_score_dict = defaultdict(lambda: defaultdict(int))
        for i, _row in lexicon_df.iterrows():
            if not pd.isnull(_row[label_column]):
                _lemma  = _row['verb'].strip()
                verb_score_dict[_lemma] = label_dict[_row[label_column]]
        
        self.verb_score_dict = verb_score_dict


    def train(self, texts, text_ids):
        self.texts = texts
        self.text_ids = text_ids
        self.persona_score_dict, \
            self.id_persona_score_dict, \
            self.id_persona_count_dict, \
            self.id_nsubj_verb_count_dict, \
            self.id_dobj_verb_count_dict, \
            self.id_persona_scored_verb_dict = self.__score_dataset(self.texts, self.text_ids)


    def get_score_totals(self):
        return dict(self.persona_score_dict)


    def get_scores_for_doc(self, doc_id):
        return dict(self.id_persona_score_dict[doc_id])


    # TODO: this would be helpful for debugging and result inspection
    # def get_docs_for_persona(self, persona):


    def get_persona_counts(self):
        return self.persona_count_dict


    def count_personas_for_doc(self, doc_id):
        return dict(self.id_persona_count_dict[doc_id])
    
    def count_scored_verbs_for_doc(self, doc_id):
        return dict(self.id_persona_scored_verb_dict[doc_id])


    def count_nsubj_for_doc(self, doc_id):
        return dict(self.id_nsubj_verb_count_dict[doc_id])


    def count_dobj_for_doc(self, doc_id):
        return dict(self.id_dobj_verb_count_dict[doc_id])
    
    
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
            isPerson = len(span.ents) > 0 and any([e.label_ in NER_TAGS for e in span.ents])
            isPerson = isPerson or any([w.text.lower()==p.lower() for w in span for p in peopleWords])
            
            if isPerson:

                # check if it's in the clusters to add people
                inClusterAlready = False
                for c in clusters:
                    if any([m.start == span.start and m.end == span.end for m in c.mentions]):
                        #print("Yes", c)      
                        peopleClusters.add(c)
                        inClusterAlready = True
                
                # also add singletons
                if not inClusterAlready:
                    #print(span)
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

        return newPeopleClusters
    

    def __parseAndExtractFrames(self, text, peopleWords=["doctor", "i", "me", "you", "he", "she", "man", "woman"]):

        nsubj_verb_count_dict = defaultdict(int)
        dobj_verb_count_dict = defaultdict(int)

        doc = nlp(text)

        clusters = self.__getPeopleClusters(doc, peopleWords=peopleWords)

        for _cluster in clusters:

            _num_non_pronouns = len([_span for _span in _cluster if str(_span).lower() not in PRONOUNS])

            # If the cluster is only third person pronouns, don't include it (coref failed to match to an entity)
            # Second condition handles weird edge case where the cluster key is in PRONOUNS but the spands include "you" or "I"
            if _num_non_pronouns > 0 and str(_cluster.main).lower() not in PRONOUNS:

                for _span in _cluster:

                    _cluster_text = str(_cluster.main).lower()
                    _cluster_text = pronoun_special_cases.get(_cluster_text, _cluster_text)

                    self.persona_count_dict[_cluster_text] += 1
                    self.entity_match_count_dict[_cluster_text][str(_span).lower()] += 1
                
                    if _span.root.dep_ == 'nsubj':
                        _verb = _span.root.head.lemma_.lower()
                        nsubj_verb_count_dict[(_cluster_text, _verb)] += 1   

                    elif _span.root.dep_ == 'dobj':
                        _verb = _span.root.head.lemma_.lower() 
                        dobj_verb_count_dict[(_cluster_text, _verb)] += 1   

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

        for (_persona, _verb), _count in dobj_verb_count_dict.items():
            if _verb in self.verb_score_dict:
                persona_scored_verbs_dict[_persona] += 1
                _theme_score = self.verb_score_dict[_verb]['theme']
                persona_score_dict[_persona] += (_count*_theme_score)

        # persona_score_dict = {p: s/float(persona_count_dict[p]) for p, s in persona_score_dict.items()} # do this later instead

        # FOR DEBUGGING
        # print('nsubj_verb_count_dict', nsubj_verb_count_dict)
        # print('dobj_verb_count_dict', dobj_verb_count_dict)
        # print('persona_count_dict', persona_count_dict)
        # print('persona_score_dict', persona_score_dict)

        return persona_score_dict, persona_scored_verbs_dict


    def __score_dataset(self, texts, text_ids):

        id_nsubj_verb_count_dict = {}
        id_dobj_verb_count_dict = {}
        id_persona_score_dict = {}
        id_persona_count_dict = {}
        id_persona_scored_verb_dict = {}

        for _text, _id in tqdm(zip(texts, text_ids), total=len(texts)):
            _nsubj_verb_count_dict, _dobj_verb_count_dict = self.__parseAndExtractFrames(_text)
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
        persona_score_dict = {p: s/float(self.persona_count_dict[p]) for p, s in persona_score_dict.items()}

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