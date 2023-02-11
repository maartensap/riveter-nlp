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

# test_getPeopleClusters()
# def test_parseAndExtractFrames():
#     # 
    
#     create_test_lex()
#     framer = ConnoFramer()
#     framer.load_lexicon("test_lex.csv", 'verb', 'agency')
#     destroy_test_lex()


#     # t = framer._ConnoFramer__parseAndExtractFrames(text)
#     # print(t)

#     text = "I accompanied my friend Brian Smith to the store, because he had abandoned his bike there. Brian also absorbs lots of complaints, so I address him as doctor. My other friend is named Brian Jones"
#     t = framer._ConnoFramer__parseAndExtractFrames(text)
#     print(t)
#     # self.__parseAndExtractFrames(_text)
#             # _nsubj_verb_count_dict, _dobj_verb_count_dict = self.__parseAndExtractFrames(_text)
#             # _persona_score_dict = self.__score_document(_nsubj_verb_count_dict, _dobj_verb_count_dict)
#             # _persona_count_dict = self.__get_persona_counts_per_document(_nsubj_verb_count_dict, _dobj_verb_count_dict)

# # example_stories = ["I was just thinking about walking down the street, when my shoelace snapped. I had to call my doctor to pick me up. I felt so bad I also called my friend Katie, who came in her car. She was a lifesaver. My friend Jack is nice.",
# #                    "My doctor fixed my shoe. I thanked him. Then Susan arrived. Now she is calling the doctor too."]
# # text_ids = [0, 1]

# # framer.train(example_stories, text_ids)

# # for x,v in framer.get_score_totals().items():
# #     print(x, v)

# test_parseAndExtractFrames()