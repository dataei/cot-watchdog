#lexicons for the hedge miscalibration detector
#these are grouped by type so that we can analyze them or weight
#them later. all matched case-insensitively, multi-word phrases
#must be matched as exact substrings (w word boundaries wherever possible)

#epistemic hedges: uncertainty abt truth
EPISTEMIC = {
    "maybe", "perhaps", "possibly", "probably", "likely",
    "might", "could", "may", "seems", "appears",
    "presumably", "supposedly", "ostensibly",
}
#self reported uncertainty: the agent admitting it doesnt know
SELF_DOUBT = {
    "i think", "i believe", "i guess", "i'd say", "i suspect",
    "my guess", "my sense", "i'm not sure", "not sure",
    "unsure", "uncertain", "i don't know", "hard to tell",
    "hard to say",
}
#numerical hedging: estimated quantities 
NUMERICAL = {
    "roughly", "approximately", "around", "about", "somewhere",
    "ish", "sort of", "kind of", "more or less", "give or take",
}
#qualified claims, which act as "softening" modifiers
QUALIFIED = {
    "tends to", "generally", "usually", "often", "sometimes",
    "in most cases", "for the most part", "by and large",
    "as a rule", "more often than not",
}
#all of the hedges combined into one set
HEDGES = EPISTEMIC | SELF_DOUBT | NUMERICAL | QUALIFIED
#confidence markers which are phrases that express certainty
DIRECT_CONFIDENCE = {
    "definitely", "certainly", "clearly", "obviously",
    "absolutely", "undoubtedly", "without a doubt",
    "no question", "no doubt",
}
ASSERTIVE_CLAIMS = {
    "the answer is", "it is", "must be", "is exactly",
    "is precisely", "is in fact", "the fact is",
}
EPISTEMIC_CERTAINTY = {
    "i'm certain", "i am certain", "i'm sure", "i am sure",
    "i know", "confirmed", "verified",
}
EVIDENTIAL_CLAIMS = {
    "proves", "demonstrates", "establishes", "shows that",
    "guarantees", "ensures",
}
CONFIDENCE_MARKERS = DIRECT_CONFIDENCE | ASSERTIVE_CLAIMS | EPISTEMIC_CERTAINTY | EVIDENTIAL_CLAIMS