#reasoning action mismatch detector
#the module is built in two parts
#extract_stated_intent: pull structured intent from English reasoning
#detect_mismatch: compare extracted intent to the actual tool call
import re
from detectors.intent_mapping import VERB_TO_TOOL
#pattern A is structured intent line
#Matches "INTENT: web_search(query)" or "TOOL: notes_write(text)"
#use this when the agent is prompted to output a structured intent line
#at the end of its reasoning
STRUCTURED_INTENT = re.compile(
    r"(?:INTENT|TOOL|NEXT)\s*:\s*(\w+)\s*\(([^)]*)\)",
    re.IGNORECASE,
)
#pattern b is verb based intent wheere it matches things like
#"ill search for x"
#"let me look up y"
#"next ill write a note"
#built dynamically from VERB_TO_TOOL so adding new verbs there 
#auto extends the pattern
_verb_alternation = "|".join(sorted(VERB_TO_TOOL.keys(), key=len, reverse=True))
VERB_INTENT = re.compile(
    r"\b(?:I'll|I will|let me|next,?\s*I'll|I'm going to|going to|"
    r"i need to|i should|my next step is to|let's)\s+"
    r"(" + _verb_alternation + r")\b",
    re.IGNORECASE,
)