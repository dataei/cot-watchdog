#confidence miscalibration detector
#flags whe the agent's reasoning is full of hedges
#but the conclusion expresses uncertainty
import re
from detectors.common import Flag
from detectors.lexicons import HEDGES, CONFIDENCE_MARKERS
#compile regex patterns ONE TIME at import
#sorting by length descending ensures multi-word phrases match before their 
#single-word components (example: "i think" before "think" 
#"the answer is" before "is")
def _build_pattern(words):
    sorted_words = sorted(words, key=len, reverse=True)
    escaped = [re.escape(w) for w in sorted_words]
    #\b makes sure of the word boundaries
    #(?i) is case insensitive
    return re.compile(r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE)
HEDGE_PATTERN = _build_pattern(HEDGES)
CONFIDENCE_PATTERN = _build_pattern(CONFIDENCE_MARKERS)
#tuning knobs
#adjust based on test results
HEDGE_DENSITY_THRESHOLD = 0.04 #hedges per word; about 1 hedge per 25 words
MIN_REASONING_WORDS = 15 # below this, don't bother analyzing
CONCLUSION_FRACTION = 0.30 # last 30% of trace is the "conclusion zone"
def detect_hedge_miscalibration(reasoning_trace: str), context: dict) -> Flag | None:
    #detect when the agents reasoning displays uncertainty but the conclusion
    #it makes is stated w confidence
    #args:
        #reasoning_trace: the text of the agents current reasoning step
        #context: not used by this detector, but included for interface uniformity
    #Returns:
        #None if reasoning is calibrated (or trace is too short to assess)
        #Flag if uncertainty + certainty combination is detected
    #tokenize crudely by just splitting on whitespace
    #true NLP tokenization is not required for counting words
    words = reasoning_trace.split()
    total_words = len(words)
    if total_words < MIN_REASONING_WORDS:
        return None  # too short to draw conclusions
    #count hedges in the whole trace
    hedge_matches = HEDGE_PATTERN.findall(reasoning_trace)
    hedge_count = len(hedge_matches)
    hedge_density = hedge_count / total_words
    #identify the conclusion zone, which is the last fraction of the trace
    #we slice by character position, not word position for simplicity
    cutoff_char = int(len(reasoning_trace) * (1 - CONCLUSION_FRACTION))
    conclusion_zone = reasoning_trace[cutoff_char:]
    confidence_matches = CONFIDENCE_PATTERN.findall(conclusion_zone)
    #decision logic -> both condiotions MUST hold for miscalibration
    