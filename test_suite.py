# Can run with:
# "pip install pytest"
# "pytest test_suite.py"

from main2 import ConnoFramer
import pandas
import os

def test_get_lemma():
    framer = ConnoFramer()
    assert framer._ConnoFramer__get_lemma_spacy("accompanies") == "accompany"
    assert framer._ConnoFramer__get_lemma_spacy("trying") == "try"

def test_load_power():
    framer = ConnoFramer()
    framer.load_lexicon('power')

    assert len(framer.verb_score_dict) == 1716
    assert framer.verb_score_dict["abolish"]["agent"] == 1
    assert framer.verb_score_dict["abolish"]["theme"] == -1
    assert framer.verb_score_dict["accompany"]["agent"] == -1
    assert framer.verb_score_dict["address"]["agent"] == 0
    assert framer.verb_score_dict["address"]["theme"] == 0


def test_load_agency():
    framer = ConnoFramer()
    framer.load_lexicon('agency')

    assert len(framer.verb_score_dict) == 2109
    assert framer.verb_score_dict["abolish"]["agent"] == 1
    assert framer.verb_score_dict["abolish"]["theme"] == 0
    assert framer.verb_score_dict["accompany"]["agent"] == 0
    assert framer.verb_score_dict["accompany"]["theme"] == 0
    assert framer.verb_score_dict["address"]["agent"] == -1
    assert framer.verb_score_dict["address"]["theme"] == 0

def test_getPeopleClusters():
    import spacy
    nlp = spacy.load('en_core_web_sm')
    import neuralcoref
    nlp.add_pipe(neuralcoref.NeuralCoref(nlp.vocab,blacklist=False),name="neuralcoref")

    text = "I accompanied my friend Brian Smith to the store, because he had abandoned his bike there."
    doc = nlp(text)
    
    framer = ConnoFramer()
    people = framer._ConnoFramer__getPeopleClusters(doc, peopleWords=["i"])

    assert len(people) == 2
    if people[0].main.text == "Brian Smith":
        assert people[1].main.text == "I"
    else:
        assert people[1].main.text == "Brian Smith"
        assert people[0].main.text == "I"



def test_parseAndExtractFrames():
    framer = ConnoFramer()

    text = "I accompanied Brian Smith to the store, because he had abandoned his bike there. Brian also absorbs lots of complaints, \
    so I address him as doctor. I also have a friend named Brian Jones. Brian Jones abuses free food."
    nsubj_verb_count_dict, dobj_verb_count_dict = framer._ConnoFramer__parseAndExtractFrames(text)
    assert len(nsubj_verb_count_dict) == 6
    assert len(dobj_verb_count_dict) == 2
    assert ("brian smith", "abandon") in nsubj_verb_count_dict
    assert ("i", "accompany") in nsubj_verb_count_dict
    assert ("brian smith", "accompany") in dobj_verb_count_dict
    assert ("brian smith", "address") in dobj_verb_count_dict

    # This is a failure case, it does not recognize that "my friend" is apposition to "Brian Jones"
    # assert ("have", "brian jones") in dobj_verb_count_dict

# Make sure it doesn't crash with empty text
def test_noChains():
    framer = ConnoFramer()

    text = "There's no tags here."
    nsubj_verb_count_dict, dobj_verb_count_dict = framer._ConnoFramer__parseAndExtractFrames(text)
    assert len(nsubj_verb_count_dict) == 0
    assert len(dobj_verb_count_dict) == 0

    text = ""
    nsubj_verb_count_dict, dobj_verb_count_dict = framer._ConnoFramer__parseAndExtractFrames(text)
    assert len(nsubj_verb_count_dict) == 0
    assert len(dobj_verb_count_dict) == 0

def test_score_document():
    framer = ConnoFramer()
    framer.load_lexicon('power')

    text = "I accompanied Brian Smith to the store, because he had abandoned his bike there. Brian also absorbs lots of complaints, \
    so I address him as doctor. I also unearthed a friend named Brian Jones. Brian Jones abuses free food."

    _nsubj_verb_count_dict, _dobj_verb_count_dict = framer._ConnoFramer__parseAndExtractFrames(text)
    persona_score_dict, persona_scored_verbs_dict  = framer._ConnoFramer__score_document(_nsubj_verb_count_dict, _dobj_verb_count_dict)

    # I: accompany, address, have
    # unearthed is not in the lexicon, address is power_equal, accompany is power_theme
    assert persona_score_dict["i"] == -1

    # accompany and address are in lexicon
    assert persona_scored_verbs_dict["i"] == 2

    # Brian Smith: dobj of accompany (power_theme +1), abandoned (power_agent +1), absorb (power_agent +1), dobj of address (equal)
    assert persona_score_dict["brian smith"] == 3
    assert persona_scored_verbs_dict["brian smith"] == 4

    # Brian Jones: abuses (power_agent +1)
    assert persona_score_dict["brian jones"] == 1
    assert persona_scored_verbs_dict["brian jones"] == 1

def test_get_persona_counts_per_document():
    framer = ConnoFramer()
    framer.load_lexicon('power')

    text = "I accompanied Brian Smith to the store, because he had abandoned his bike there. Brian also absorbs lots of complaints, \
    so I address him as doctor. I also have a friend named Brian Jones. Brian Jones abuses free food."

    _nsubj_verb_count_dict, _dobj_verb_count_dict = framer._ConnoFramer__parseAndExtractFrames(text)
    persona_count_dict = framer._ConnoFramer__get_persona_counts_per_document(_nsubj_verb_count_dict, _dobj_verb_count_dict)

    # I: accompany, address, have
    assert persona_count_dict["i"] == 3
    # Brian Smith: dobj of accompany (power_theme +1), abandoned (power_agent +1), absorb (power_agent +1), dobj of address (equal)
    assert persona_count_dict["brian smith"] == 4

    # Brian Jones: abuses (power_agent +1)
    assert persona_count_dict["brian jones"] == 1

# Other components test parts of the train pipeline, this one tests it in full
# It also inadverntly tests having pronoun direct objects (e.g. me)
def test_train():
    framer = ConnoFramer()
    framer.load_lexicon('power')

    texts = ["I accompanied Brian Smith to the store, because he had abandoned his bike there. Brian also absorbs lots of complaints, \
    so I address him as doctor. I also have a friend named Brian Jones. Brian Jones abuses free food.",
    "Brian Smith accompanies me"]

    framer.train(texts, [0,1])
    persona_score_dict = framer.get_score_totals()

    # These are same values as previous test with the added sentence: Brian Smith accompanies me, which is +1 i and -1 brian smith
    assert persona_score_dict["i"] == 0
    assert persona_score_dict["brian smith"] == 2
    assert persona_score_dict["brian jones"] == 1

    persona_count_dict_1 = framer.count_personas_for_doc(0)
    assert persona_count_dict_1["i"] == 3
    assert persona_count_dict_1["brian smith"] == 4
    assert persona_count_dict_1["brian jones"] == 1
        
    persona_count_dict_1 = framer.count_scored_verbs_for_doc(0)
    assert persona_count_dict_1["i"] == 2 # have is not in lex

    persona_count_dict_2 = framer.count_personas_for_doc(1)
    assert persona_count_dict_2["i"] == 1
    assert persona_count_dict_2["brian smith"] == 1
    assert "brian jones" not in persona_count_dict_2
    

    persona_score_dict_1 = framer.get_scores_for_doc(0)
    assert persona_score_dict_1["i"] == -1
    assert persona_score_dict_1["brian smith"] == 3
    assert persona_score_dict_1["brian jones"] == 1

    persona_score_dict_2 = framer.get_scores_for_doc(1)
    assert persona_score_dict_2["i"]== 1
    assert persona_score_dict_2["brian smith"] == -1
    assert "brian jones" not in persona_score_dict_2

    nsubj_doc1 = framer.count_nsubj_for_doc(0)
    dobj_doc1 = framer.count_dobj_for_doc(0)
    assert len(nsubj_doc1) == 6
    assert len(dobj_doc1) == 2
    assert ("brian smith", "abandon") in nsubj_doc1
    assert ("i", "accompany") in nsubj_doc1
    assert ("brian smith", "accompany") in dobj_doc1
    assert ("brian smith", "address") in dobj_doc1

    nsubj_doc2 = framer.count_nsubj_for_doc(1)
    dobj_doc2 = framer.count_dobj_for_doc(1)
    assert len(nsubj_doc2) == 1
    assert len(dobj_doc2) == 1
    assert ("brian smith", "accompany") in nsubj_doc2
    assert ("i", "accompany") in dobj_doc2

def test_evaluate_verb_coverage():
    # TODO
    pass

def test_people_noun_chunk():
    # This is the example that was causing a crash, noun chunk is the last value in string
    text = "Hassan worked hard and quickly rose through the ranks. Hassan"
    framer = ConnoFramer()
    framer._ConnoFramer__parseAndExtractFrames(text)

# Check that "I" is getting considered as a person (even though it is capitalized)
def test_find_people():
    text = "My name is Francis and I am originally from Vietnam. I came to America when I was just a young man."

    framer = ConnoFramer()
    subj_verb_count_dict, dobj_verb_count_dict = framer._ConnoFramer__parseAndExtractFrames(text)
    assert len(subj_verb_count_dict) == 2


# This uses the same example as the demo. Use it to make sure demo isn't broken
def test_demo():
    example_stories = ["I was just thinking about walking down the street, when my shoelace snapped. I had to call my doctor to pick me up. I felt so bad I also called my friend Katie, who came in her car. She was a lifesaver. My friend Jack is nice.",
                   "My doctor fixed my shoe. I thanked him. Then Susan arrived. Now she is calling the doctor too."]
    text_ids = [0, 1]
    framer = ConnoFramer()
    framer.load_sap_lexicon('power')
    framer.train(example_stories, text_ids)

    # In the second document, "I" should get mapped to "i" instead of "my"
    assert ('i', 'thank') in framer.id_nsubj_verb_count_dict[1]
    assert ('my', 'thank') not in framer.id_nsubj_verb_count_dict[1]

    # I think (0), I have (+1), I feel (0), I call (-1), pick me (-1), but "have" doesn't lemmatize
    assert framer.id_persona_score_dict[0]["i"] == -2
    assert framer.id_persona_scored_verb_dict[0]["i"] == 4

    # I thank (-1)
    assert framer.id_persona_score_dict[1]["i"] == -1
    # print(framer.id_persona_scored_verb_dict[0]["i"])
    # print(framer.id_persona_scored_verb_dict[1]["i"])
    score_totals = framer.get_score_totals()
    assert score_totals["i"] == -3