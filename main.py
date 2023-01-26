#!/usr/bin/env python3

import numpy as np
import pandas as pd
import argparse

from IPython import embed

def loadFile(input_file):
  """@todo: make this read in multiple types of files"""
  df = pd.read_csv(input_file)
  return df

#### Entity parsing
#todo, will be split off into its own file probably
import spacy
nlp = spacy.load('en_core_web_sm')#,parse=True,entity=True,tag=True)

import neuralcoref
nlp.add_pipe(neuralcoref.NeuralCoref(nlp.vocab,blacklist=False),name="neuralcoref")
# neuralcoref.add_to_pipe(nlp) 

ner_tags = ["PERSON"]

def getCorefClusters(spacyDoc):
  clusters = spacyDoc._.coref_clusters
  # nvm this this was a flag in the neural coref
  # adding the "first person" cluster (you / i), for some reason that didn't work
  # i = ["I","my","mine","me"] # possibly, lowercase??
  # you = ["you","yours","your","You","Yours","Your"]
  # newClusters = {}
  # for t in spacyDoc:
  #   if str(t) in i: 
  #     newClusters["I"] = newClusters.get("I",[])
  #     newClusters["I"].append(t)
  #     embed();exit()
  #   if str(t) in you: 
  #     newClusters["you"] = newClusters.get("you",[])
  #     newClusters["you"].append(t)
      
  # adding them to the clusters
  # for main,mentions in newClusters.items():
  #   clusters.append(neuralcoref.neuralcoref.Cluster(len(clusters),main,mentions))
  return clusters

def getPeopleClusters(spacyDoc,peopleWords=["doctor"]):
  clusters = getCorefClusters(spacyDoc)

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

  
def findVerbPhrases(spacyDoc):
  # @todo, make this better, remove auxiliary verbs ? or merge them?
  verbTags = ["VERB","AUX"]
  verbs = [t for t in spacyDoc if t.pos_ in verbTags]

  return verbs

def parseAndExtractFrames(text,peopleWords=["doctor"]):
  doc = nlp(text)

  # coref clusters
  clusters = getPeopleClusters(doc,peopleWords=peopleWords)

  # get verbs
  verbs = findVerbPhrases(doc)
  
  # matching people mentions to verbs
  mention2verbs = {}
  
  verbTags = ["VERB","AUX"]

  # maarten's attempt but I forgot to keep track of which direction the mention attaches to the verb
  # so I'm giving up for tonight
  # for c in clusters:
  #   mention2verbs[c.main] = []
  #   for m in c:
  #     # climb the tree to get to the verb?
  #     for a in m.root.ancestors:
  #       if a.pos_ in verbTags:
  #         mention2verbs[c.main].append((m,a))
  #         embed();exit()
  #         break
          

      
  people2verbs = {}

def main(args):
  df = loadFile(args.input_file)

  out = df[args.text_column].apply(parseAndExtractFrames)
  

if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument("-i","--input_file")
  p.add_argument("-o","--output_file")
  p.add_argument("-c","--text_column",default="text")
  args = p.parse_args()
  main(args)
