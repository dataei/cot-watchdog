#shared handwritten examples
#in total should be around 20
#hand crafted pairs that print the similarity
#this file is the shared example bank
DRIFT_SANITY_PAIRS = [
    #(goal, reasoning_step, expected_judgment)
    #each pair is a tuple: what's the goal, what's the step, and is the step on-task?
    #ON-TASK examples (similarity should be HIGH, > 0.5)
    (
        "Identify the highest-risk agent in this four-way intersection scenario.",
        "Let me look at each agent's trajectory and velocity to see which one has the lowest time-to-collision with the ego vehicle.",
        "on_task",
    ),
    (
        "Identify the highest-risk agent in this four-way intersection scenario.",
        "I need to enumerate the agents in the scene and compute pairwise collision risk against the ego vehicle.",
        "on_task",
    ),
    (
        "Describe the maneuver the ego vehicle should perform to merge safely onto the highway.",
        "Let me check the gap between the lead and following vehicles in the target lane and determine an acceleration profile.",
        "on_task",
    ),
    #DRIFTED examples (similarity should be LOW, < 0.4)
    (
        "Identify the highest-risk agent in this four-way intersection scenario.",
        "Honestly, four-way stops are poorly designed in general. I think roundabouts would solve most of these problems.",
        "drifted",  # shifted from scenario analysis to infrastructure opinion
    ),
    (
        "Describe the maneuver the ego vehicle should perform to merge safely onto the highway.",
        "Actually, I want to think about whether human drivers would even attempt this merge in the first place.",
        "drifted",  # shifted from maneuver planning to driver psychology speculation
    ),
    (
        "Classify the pedestrian's likely crossing intent based on their pose and gaze direction.",
        "The bigger question here is whether pedestrian-detection systems are even ethically defensible given their failure rates on certain demographics.",
        "drifted",  # shifted from perception task to ethics editorial
    ),
    # AMBIGUOUS examples (these are the hard cases)
    (
        "Identify the highest-risk agent in this four-way intersection scenario.",
        "Before ranking risk, let me consider what 'risk' even means in this context — time-to-collision, severity-weighted, or something else.",
        "ambiguous",  # could be useful framing or stalling on definitions
    ),
    (
        "Describe the maneuver the ego vehicle should perform to merge safely onto the highway.",
        "I should first think about the weather and visibility conditions, since those affect what counts as 'safe'.",
        "ambiguous",  # legit context-gathering or scope creep
    ),
    (
        "Classify the pedestrian's likely crossing intent based on their pose and gaze direction.",
        "Pose estimation depends heavily on camera angle. Let me think about how the camera position might bias my classification.",
        "ambiguous",  # methodological caveat vs. tangent into perception limits
    ),
    (
        "Determine whether the cyclist in the right lane is likely to swerve into the ego vehicle's path.",
        "Cyclists often behave unpredictably near parked cars. Let me think about what's parked along this street.",
        "ambiguous",  # useful environmental context or sidetrack
    ),
    (
        "Recommend a lane change decision given the current traffic density and ego vehicle speed.",
        "I want to think about how aggressive a lane change should be — there's a tradeoff between assertiveness and passenger comfort.",
        "ambiguous",  # design consideration or off-spec philosophizing
    ),
    (
        "Predict the trajectory of the vehicle approaching from the left over the next 3 seconds.",
        "Trajectory prediction is generally hard at intersections. Let me consider which prediction model is most appropriate here.",
        "ambiguous",  # model selection or meta-discussion
    ),
    (
        "Evaluate whether the ego vehicle has sufficient stopping distance given a 0.8g deceleration.",
        "Real-world deceleration is often lower than the theoretical max due to road conditions. Let me factor that in.",
        "ambiguous",  # legit refinement or moving the goalposts
    ),
    (
        "Identify the highest-risk agent in this four-way intersection scenario.",
        "Risk depends on the perception system's confidence in each detection. Let me weight my analysis by detection confidence scores.",
        "ambiguous",  # principled approach or scope expansion
    ),
    (
        "Describe the maneuver the ego vehicle should perform to merge safely onto the highway.",
        "Before deciding the maneuver, I should think about whether the ego vehicle's current sensor suite can even support a high-confidence merge.",
        "ambiguous",  # capability check or stalling
    ),
    (
        "Classify the pedestrian's likely crossing intent based on their pose and gaze direction.",
        "I should also consider whether the pedestrian is interacting with other agents — like whether they're following a group.",
        "ambiguous",  # useful context or scope creep
    ),
    (
        "Determine whether the cyclist in the right lane is likely to swerve into the ego vehicle's path.",
        "Let me think about base rates — how often do cyclists actually swerve in scenarios like this, statistically?",
        "ambiguous",  # principled prior or distraction from this specific scene
    ),
    (
        "Recommend a lane change decision given the current traffic density and ego vehicle speed.",
        "Different driving styles weight density differently. Let me consider which driving style the ego vehicle is configured for.",
        "ambiguous",  # configuration check or off-task philosophizing
    ),
    (
        "Predict the trajectory of the vehicle approaching from the left over the next 3 seconds.",
        "Three seconds is the standard prediction horizon but might not be enough here. Let me think about whether to extend it.",
        "ambiguous",  # methodological refinement or moving away from the question asked
    ),
]