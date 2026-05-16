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
