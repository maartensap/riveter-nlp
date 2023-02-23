Folder created by Hannah Rashkin (hrashkin@cs.washington.edu) in November 2015
Last Updated: May 4, 2016
==============================================================================

DOCUMENT DESCRIPTION:
The connotation frames are listed in the "full_frame_info.txt" file, which is a tab-separated file with the ratings created from crowdsourced annotations.  These annotations were created in November 2015, via AMT crowdsourcing.

The first line of the document is a header with the labels for the different fields represented by each column of the file.  In the notation used in the labels: s=subject, o=object, w=writer, r=reader.  So, for example, "Perspective(so)" is the perspective from the subject towards the object.

In the remaining lines, each line contains annotations for a single verb predicate.  

The entire file includes ~940 verbs with 12 aspect ratings per verb.  The first 9 aspects [perspective(wo), perspective(ws), perspective(so), effect(o), effect(s), value(o), value(s), state(o),state(s)] are included in our experiments discussed in our paper.  The additional 3 aspects [perspective(os), perspective(ro), perspective(rs)] are ratings that were retrieved from annotator responses but not used in the paper.

All scores are in the range [-1.0, 1.0].  For our purposes, we treated the cut-offs as:
[-1.0, -0.25) :	-/Negative
[-.25, 0.25] :	=/Neutral
(0.25, 1.0] :	+/Positive

------------------------------------------------------------------------------

CROWD SOURCING PROCESS: 
There were 15 annotations per verb that were averaged together to create the ratings in the file.  When taking the average, AMT responses that were between Positive and Negative were mapped to numeric intervals:
"Positive": 1.0
"Positive or Neutral": 0.5
"Neutral": 0.0
"Negative or Neutral": -0.5
"Negative": -1.0

Because of this, 0.25 was selected as a cutoff for the final averaged ratings since it falls halfway between the score for "Neutral" and the score for "Positive or Neutral"/"Negative or Neutral".

More details can be found in the paper "Connotation Frames: A Data-Driven Investigation" (in submission).
