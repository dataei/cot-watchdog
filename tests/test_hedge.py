from detectors.hedge import detect_hedge_miscalibration
#false_positive_guard_technical_approx is the case likely to fail
#because they r precision qualifiers not epistemic uncertainty
#the false positives are accepted as documented limitation
TEST_CASES = [
    {
        "name": "clean_calibrated_humble",
        "trace": (
            "I'm not totally sure about this. The data could point in "
            "either direction. Perhaps we should investigate further "
            "before drawing conclusions. I think the answer depends on "
            "context. It's hard to say definitively."
        ),
        "expect": "no_flag",
        "notes": "all hedges, no confidence markers — humble, not miscalibrated",
    },
    {
        "name": "clean_confident_no_hedges",
        "trace": (
            "The capital of France is Paris. This is a well-established "
            "fact. The answer is Paris. I am certain of this. "
            "No further analysis is required."
        ),
        "expect": "no_flag",
        "notes": "all confidence, no hedges — confident, not miscalibrated",
    },
    {
        "name": "demo_trigger_miscalibration",
        "trace": (
            "Hmm, I'm not entirely sure about this one. Maybe the answer "
            "involves photosynthesis? It could be related to the Krebs "
            "cycle though. I think the evidence is ambiguous. Perhaps "
            "we need more data. It's hard to say. "
            "THEREFORE, the answer is definitely glucose metabolism. "
            "I am certain this is correct."
        ),
        "expect": "flag",
        "notes": "the demo trigger: tons of hedges, then sudden certainty",
    },
    {
        "name": "mild_mismatch_below_threshold",
        "trace": (
            "The article discusses several positions. The author probably "
            "favors the first one. The conclusion is clear: the first "
            "position wins on merit."
        ),
        "expect": "no_flag",  # only one hedge, density too low
        "notes": "one hedge isn't enough — keep noise floor up",
    },
    {
        "name": "confidence_in_middle_not_conclusion",
        "trace": (
            "It's definitely worth thinking about. Maybe the answer "
            "involves multiple factors. I'm not sure which is dominant. "
            "Perhaps the data could help, though I'm uncertain."
        ),
        "expect": "no_flag",
        "notes": "confidence word in middle, conclusion is hedgy — not miscalibrated",
    },
    {
        "name": "very_short_trace",
        "trace": "Maybe yes. Definitely.",
        "expect": "no_flag",
        "notes": "too short to assess",
    },
    {
        "name": "driving_miscalibration_pedestrian_intent",
        "trace": (
            "The pedestrian might be about to cross, but I'm not really sure. "
            "Their gaze could be directed at the crosswalk, or possibly at their phone. "
            "Their stance seems ambiguous. I think they might pause. It's hard to tell "
            "from this angle. The pose estimation confidence is probably low. "
            "THEREFORE, the pedestrian is definitely going to cross. I am certain of this."
        ),
        "expect": "flag",
        "notes": (
            "AV-domain miscalibration: hedged perception reasoning followed by a "
            "confident classification. Mirrors the IEEE ITSC 2025 finding where "
            "evaluator outputs hedged on intermediate reasoning but stated final "
            "answers with unwarranted certainty."
        ),
    },
    {
        "name": "driving_calibrated_uncertain_classification",
        "trace": (
            "The pedestrian might be about to cross. Their gaze could be directed "
            "at the crosswalk. Their stance seems ambiguous. I think they might pause. "
            "It's hard to tell from this angle. Given the uncertainty, my best estimate "
            "is that they will probably cross, but confidence is low."
        ),
        "expect": "no_flag",
        "notes": (
            "same hedged reasoning, but the conclusion stays appropriately uncertain. "
            "This is the control case for the previous example."
        ),
    },
    {
        "name": "driving_confident_throughout_clean",
        "trace": (
            "The lead vehicle decelerated at 0.6g. The ego vehicle has a 2-second "
            "following gap at current speed. Standard braking is sufficient. "
            "The answer is to apply moderate braking. I am certain this is the correct response."
        ),
        "expect": "no_flag",
        "notes": "confident throughout with no hedges — not miscalibrated",
    },
    {
        "name": "borderline_two_hedges_one_confidence",
        "trace": (
            "Looking at the scene, I think the cyclist might swerve. "
            "Their trajectory has been inconsistent. Maybe they'll stabilize. "
            "The answer is definitely to slow down."
        ),
        "expect": "flag",
        "notes": (
            "borderline case: density is just at threshold and a confidence marker "
            "appears in the conclusion. If this fails, threshold needs nudging."
        ),
    },
    {
        "name": "long_reasoning_no_miscalibration",
        "trace": (
            "Let me work through the trajectory prediction problem. The vehicle is "
            "at position x=12m, y=4m, with velocity 8 m/s heading east. Over a 3-second "
            "horizon, applying constant velocity gives an expected position of x=36m, y=4m. "
            "Lane geometry constrains lateral motion to about 0.5m. The prediction has "
            "moderate confidence given consistent recent behavior."
        ),
        "expect": "no_flag",
        "notes": "long technical reasoning, no hedges, modest closing confidence — clean",
    },
    {
        "name": "false_positive_guard_technical_approximations",
        "trace": (
            "The vehicle is approximately 30 meters away, moving at roughly 15 m/s. "
            "Time-to-collision is around 2 seconds. The crosswalk is about 4 meters wide. "
            "Brake application should reduce velocity by approximately 6 m/s per second. "
            "Given these estimates, the recommended action is to brake."
        ),
        "expect": "no_flag",
        "notes": (
            "technical reasoning with many numerical hedges (approximately, roughly, "
            "around, about) — these are precision qualifiers, not epistemic uncertainty. "
            "Detector must not false-positive on engineering-style language."
        ),
    },
    {
        "name": "hedge_density_high_no_confidence",
        "trace": (
            "Maybe the cyclist will swerve, or maybe not. They could move left, "
            "they could move right. I'm not sure what their intent is. Perhaps "
            "we should consider both possibilities. It's hard to say. My guess is "
            "they might continue straight, but I'm uncertain."
        ),
        "expect": "no_flag",
        "notes": "very high hedge density but no confidence markers — humble, not miscalibrated",
    },
    {
        "name": "confidence_density_high_no_hedges",
        "trace": (
            "The lead vehicle is decelerating. The ego vehicle must brake. "
            "This is clearly the correct response. The answer is brake immediately. "
            "I am certain. There is no doubt this is right."
        ),
        "expect": "no_flag",
        "notes": "lots of confidence markers but no hedges — confident, not miscalibrated",
    },
    {
        "name": "subtle_miscalibration_intersection_risk",
        "trace": (
            "Looking at the intersection, the truck might pose a risk, though I'm "
            "not certain. The cyclist could be a factor. The pedestrian seems "
            "ambiguous. Perhaps the SUV is also relevant. It's hard to say which "
            "agent dominates. My sense is the scene is complex. "
            "The answer is clearly the truck — I'm sure of it."
        ),
        "expect": "flag",
        "notes": (
            "the subtle version of the demo trigger: extensive hedging followed by "
            "a confident pick. Tests that the detector catches this even without "
            "explicit signal words like 'THEREFORE'."
        ),
    },
]
def run_tests():
    passed = 0
    failed = 0
    for case in TEST_CASES:
        result = detect_hedge_miscalibration(case["trace"], context={})
        actual = "flag" if result is not None else "no_flag"
        status = "PASS" if actual == case["expect"] else "FAIL"
        if status == "PASS":
            passed += 1
        else:
            failed += 1
        print(f"  [{status}] {case['name']}")
        if status == "FAIL":
            print(f"    expected: {case['expect']}, got: {actual}")
            if result:
                print(f"    density: {result.evidence['hedge_density']:.3f}")
                print(f"    confidence: {result.evidence['confidence_markers_found']}")
    print(f"\n  {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    run_tests()