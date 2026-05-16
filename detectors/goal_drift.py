#file setup
#compares agent's current reasoning step to its 
#original stated goal, using sentence embeddings
#and cosine similarity, flags when the reasoning has diverged
#from the goal beyond a threshold
import numpy as np
from sentence_transformers import SentenceTransformer
from detectors.common import Flag
#load the embedding model once at module import, avoids
#reloading the 80MB model on every detection call
_MODEL = SentenceTransformer("all-mpnet-base-v2")
#threshold below which a single step is considered drifted
#tuned against sanity pairs in examples/cot_samples.py
GOAL_DRIFT_THRESHOLD = 0.45  # <- update with our calibrated value
# How many recent steps to look at when deciding "sustained" drift
SUSTAINED_WINDOW = 3
#main function
def detect_goal_drift(reasoning_trace: str, context: dict) -> Flag | None:
    #check if current reasoning step has drifted from original goal
    #args:
        #reasoning_trace: the text of the agent's current reasoning step
        #context: dict carrying state across steps. expected keys:
            #"original_goal" (str): the task the agent was given at step 0
            #"history" (list[dict]): records from prior steps. each record
            #has at least a similarity key w the float similarity score
            #from that step. empty list on step 1
    #returns: 
        #none if the step looks on track
        #flag if drift is detected
    goal = context["original_goal"]
    history = context.get("history", [])
    #embed both texts
    #normalize_embeddings=True means we can use np.dot directly as cosine similarity
    #(no need to divide by norms)
    goal_vec = _MODEL.encode(goal, normalize_embeddings=True)
    step_vec = _MODEL.encode(reasoning_trace, normalize_embeddings=True)
    similarity = float(np.dot(goal_vec, step_vec))
    #case 1: similarity is above threshold, no flag
    if similarity >= GOAL_DRIFT_THRESHOLD:
        return None
    #case 2: similarity is below threshold. decide severity
    #severity scales with how far below threshold we are, capped at 1.0
    #a similarity of 0.45 -> severity 0.55. a similarity of 0.10 -> severity 0.90
    base_severity = 1.0 - similarity
    