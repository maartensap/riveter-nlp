# Can run with:
# "pip install pytest"
# "pytest test_suite.py"

from main2 import ConnoFramer
import pandas
import os

def create_test_lex():
    test_lex = ["verb,agency,power", "abandons,agency_pos,power_agent", "abolishes,agency_pos,power_agent", "absorbs,agency_pos,power_agent",
    "abuses,agency_pos,power_agent", "accompanies,agency_equal,power_theme", "addresses,agency_neg,power_equal"]
    with open("test_lex.csv", "w") as test_lex_filename:
        for n in test_lex:
            test_lex_filename.write(n + "\n")

def destroy_test_lex():
    os.remove("test_lex.csv")

def test_get_lemma():
    framer = ConnoFramer()
    assert framer._ConnoFramer__get_lemma_spacy("accompanies") == "accompany"
    assert framer._ConnoFramer__get_lemma_spacy("trying") == "try"

def test_load_power():
    create_test_lex()
    framer = ConnoFramer()
    framer.load_lexicon("test_lex.csv", 'verb', 'power')
    destroy_test_lex()

    assert len(framer.verb_score_dict) == 6
    assert framer.verb_score_dict["abolish"]["agent"] == 1
    assert framer.verb_score_dict["abolish"]["theme"] == 0
    assert framer.verb_score_dict["accompany"]["agent"] == 0
    assert framer.verb_score_dict["address"]["agent"] == 0
    assert framer.verb_score_dict["address"]["theme"] == 0


def test_load_agency():
    create_test_lex()
    framer = ConnoFramer()
    framer.load_lexicon("test_lex.csv", 'verb', 'agency')
    destroy_test_lex()

    assert len(framer.verb_score_dict) == 6
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
    people = framer._ConnoFramer__getPeopleClusters(doc)

    assert len(people) == 2
    assert people[0].main.text == "Brian Smith"
    assert people[1].main.text == "I"

def test_parseAndExtractFrames():
    framer = ConnoFramer()

    text = "I accompanied Brian Smith to the store, because he had abandoned his bike there. Brian also absorbs lots of complaints, so I address him as doctor. I also have a friend named Brian Jones. Brian Jones abuses free food."
    nsubj_verb_count_dict, dobj_verb_count_dict = framer._ConnoFramer__parseAndExtractFrames(text)
    assert len(nsubj_verb_count_dict) == 6
    assert len(dobj_verb_count_dict) == 2
    assert ("brian smith", "abandon") in nsubj_verb_count_dict
    assert ("i", "accompany") in nsubj_verb_count_dict
    assert ("brian smith", "accompany") in dobj_verb_count_dict
    assert ("brian smith", "address") in dobj_verb_count_dict

    # This is a failure case, it does not recognize that "my friend" is apposition to "Brian Jones"
    # assert ("have", "brian jones") in dobj_verb_count_dict

def test_score_document():
    # TODO
    pass

def test_score_document():
    # TODO
    pass

def test_get_persona_counts_per_document():
    # TODO
    pass

def test_train():
    # TODO
    pass

def test_get_persona_counts_per_document():
    # TODO
    pass

def test_evaluate_verb_coverage():
    # TODO
    pass