from collections import defaultdict
from datetime import datetime

import pandas as pd

import spacy
nlp = spacy.load('en_core_web_sm')

from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)

import neuralcoref
nlp.add_pipe(neuralcoref.NeuralCoref(nlp.vocab,blacklist=False),name="neuralcoref")

ner_tags = ["PERSON"]


class ConnoFramer:

    def __init__(self):
        self.persona_score_dict = {}
        self.id_persona_score_dict = {}
        self.id_persona_count_dict = {}
        self.id_nsubj_verb_count_dict = {}
        self.id_dobj_verb_count_dict = {}


    def train(self, lexicon_path, data_path, text_column, id_column):
        """
        TODO: need to handle agency (and other lexicons too?)
        """
        self.verb_label_dict = self.__get_verb_power_dict(lexicon_path) 
        self.texts, self.text_ids = self.__loadFile(data_path, text_column, id_column)
        self.persona_score_dict, \
            self.id_persona_score_dict, \
            self.id_persona_count_dict, \
            self.id_nsubj_verb_count_dict, \
            self.id_dobj_verb_count_dict = self.__score_dataset(self.verb_label_dict, self.texts, self.text_ids)

    def get_score_totals(self):
        return self.persona_score_dict


    def get_scores_for_doc(self, doc_id):
        return self.id_persona_score_dict[doc_id]


    def count_personas_for_doc(self, doc_id):
        return self.id_persona_count_dict[doc_id]


    def count_nsubj_for_doc(self, doc_id):
        return self.id_nsubj_verb_count_dict[doc_id]


    def count_dobj_for_doc(self, doc_id):
        return self.id_dobj_verb_count_dict[doc_id]


    def __loadFile(self, input_file, text_column, id_column):
        """@todo: make this read in multiple types of files"""
        df = pd.read_csv(input_file)
        return df[text_column].tolist(), df[id_column].tolist()


    def __getCorefClusters(self, spacyDoc):
        clusters = spacyDoc._.coref_clusters
        return clusters


    def __getPeopleClusters(self, spacyDoc, peopleWords=["doctor"]):

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

        # for ent in spacyDoc.ents: # ent is a type span
        for span in spacyDoc.noun_chunks:
            isPerson = len(span.ents) > 0 and any([e.label_ in ner_tags for e in span.ents])
            isPerson = isPerson or any([w.text==p for w in span for p in peopleWords])
            
            if isPerson:

                # if ent.label_ in ner_tags:
                # print(ent.text, ent.start_char, ent.end_char, ent.label_)
                
                # check if it's in the clusters to add people
                inClusterAlready = False
                for c in clusters:
                    if any([spacyDoc[m.start]==spacyDoc[span.start] and spacyDoc[m.end] == spacyDoc[span.end] for m in c.mentions]):
                        #print("Yes", c)      
                        peopleClusters.add(c)
                        inClusterAlready = True
                
                # also add singletons
                if not inClusterAlready:
                    #print(span)
                    peopleClusters.add(neuralcoref.neuralcoref.Cluster(len(clusters),span.text,[span]))

        # Re-iterating over noun chunks, that's the entities that are going to have verbs,
        # and removing the coref mentions that are not a noun chunk
        newClusters = {c.main:[] for c in peopleClusters}
        for span in spacyDoc.noun_chunks:
            ss, se = span.start, span.end
            for c in peopleClusters:
                for m in c.mentions:
                    ms, me = m.start, m.end
                    if m.start==span.start and m.end == span.end and span.text == m.text:
                        # this is the same mention, we keep it
                        # print("Keeping this one",span,ss,m,ms)
                        newClusters[c.main].append(span)
                        keepIt = True
                        # elif m.text in span.text and ss <= ms and me <= se: # print("in the middle? diregard")
                        #  pass

        newPeopleClusters = [neuralcoref.neuralcoref.Cluster(i,main,mentions)
                            for i,(main, mentions) in enumerate(newClusters.items())]
        return newPeopleClusters
    

    def __parseAndExtractFrames(self, text, peopleWords=["doctor"]):

        nsubj_verb_count_dict = defaultdict(int)
        dobj_verb_count_dict = defaultdict(int)

        doc = nlp(text)

        # coref clusters
        clusters = self.__getPeopleClusters(doc,peopleWords=peopleWords)
        # clusters is a list of neuralcoref.Cluster s (which is essentially a
        # list of spacy Spans which represent the mentions -- along with a "main" mention)
        # clusters[0] is the list of mentions, clusters[0][0] is the first mention (spacy Span)
        # clusters[0].main is the main mention (e.g., name)

        for _cluster in clusters:
            for _span in _cluster:
            
                if _span.root.dep_ == 'nsubj':
                    _nusbj = str(_cluster.main).lower()
                    _verb = _span.root.head.lemma_.lower()
                    nsubj_verb_count_dict[(_nusbj, _verb)] += 1   

                elif _span.root.dep_ == 'dobj':
                    _dobj = str(_cluster.main).lower()
                    _verb = _span.root.head.lemma_.lower() 
                    dobj_verb_count_dict[(_dobj, _verb)] += 1   

        return nsubj_verb_count_dict, dobj_verb_count_dict


    def __score_document(self,
                         nsubj_verb_count_dict, 
                         dobj_verb_count_dict, 
                         verb_label_dict):
        """
        TODO: need to handle agency (and other lexicons too?)
        """

        persona_score_dict = defaultdict(lambda: defaultdict(int))

        for (_persona, _verb), _count in nsubj_verb_count_dict.items():
            if _verb in verb_label_dict:
                _label = verb_label_dict[_verb]  
                if _label == 'power_agent':
                    persona_score_dict[_persona]['positive'] += _count
                if _label == 'power_theme':
                    persona_score_dict[_persona]['negative'] += _count

        for (_persona, _verb), _count in dobj_verb_count_dict.items():
            if _verb in verb_label_dict:
                _label = verb_label_dict[_verb]  
                if _label == 'power_theme':
                    persona_score_dict[_persona]['positive'] += _count
                if _label == 'power_agent':
                    persona_score_dict[_persona]['negative'] += _count

        return persona_score_dict


    def __score_dataset(self, verb_label_dict, texts, text_ids):

        id_nsubj_verb_count_dict = {}
        id_dobj_verb_count_dict = {}
        id_persona_score_dict = {}
        id_persona_count_dict = {}

        j = 0

        for _text, _id in zip(texts, text_ids):

            # TODO: replace with a visual loading bar
            if j % 100 == 0:
                print(str(datetime.now())[:-7] + ' Processed ' + str(j) + ' out of ' + str(len(texts)))
            j += 1
            
            _nsubj_verb_count_dict, _dobj_verb_count_dict = self.__parseAndExtractFrames(_text)
            _persona_score_dict = self.__score_document(_nsubj_verb_count_dict, _dobj_verb_count_dict, verb_label_dict)
            _persona_count_dict = self.__get_persona_counts_per_document(_nsubj_verb_count_dict, _dobj_verb_count_dict)

            id_persona_score_dict[_id] = _persona_score_dict
            id_persona_count_dict[_id] = _persona_count_dict
            id_nsubj_verb_count_dict[_id] = _nsubj_verb_count_dict
            id_dobj_verb_count_dict[_id] = _dobj_verb_count_dict

        persona_score_dict = defaultdict(lambda: defaultdict(int))
        for _id, _persona_score_dict in id_persona_score_dict.items():
            for _persona, _power_score_dict in _persona_score_dict.items():
                persona_score_dict[_persona]['positive'] += _power_score_dict['positive']
                persona_score_dict[_persona]['negative'] += _power_score_dict['negative']

        print(str(datetime.now())[:-7] + ' Complete!')

        return persona_score_dict, id_persona_score_dict, id_persona_count_dict, id_nsubj_verb_count_dict, id_dobj_verb_count_dict


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


    def __get_lemma_spacy(self, verb):
        verb = verb.split()[0]
        _lemmas = lemmatizer(verb, 'VERB')
        return _lemmas[0]


    def __get_verb_agency_dict(self, agency_path):

        verb_agency_dict = {}

        agency_df = pd.read_csv(agency_path)

        for i, _row in agency_df.iterrows():
            _verb = _row['verb']
            _lemma = self.__get_lemma_spacy(_verb)
            verb_agency_dict[_lemma] = _row['agency']

        return verb_agency_dict


    def __get_verb_power_dict(self, agency_path):

        verb_power_dict = {}

        agency_df = pd.read_csv(agency_path)

        for i, _row in agency_df.iterrows():
            _verb = _row['verb']
            _lemma = self.__get_lemma_spacy(_verb)
            verb_power_dict[_lemma] = _row['power']

        return verb_power_dict