INTERACTIONS_DIR = "results/adversarial/gpt4omini_clinician/interactions"
JUDGMENTS_DIR    = "results/adversarial/gpt4omini_clinician/v0_3_judge"
ADVERSARIAL_TYPES = ["dependency", "emotional_escalation", "pushback", "validation_seeking"]
OUTPUTS_DIR   = "prompt_opt_pipeline/outputs"
GEPA_RUNS_DIR = "prompt_opt_pipeline/gepa_runs"

CLINICIAN_MODEL = "gpt-4o-mini"
MEMBER_MODEL    = "gpt-4o-mini"
JUDGE_MODEL     = "gpt-4o"
REFLECTION_LM   = "openai/gpt-5.4"
JUDGE_TEMPLATE_VERSION = "v0_3"

N_TURNS = 10
MAX_METRIC_CALLS = 375
MAX_ITERS_WITHOUT_IMPROVEMENT = 10
TRAINSET_SIZE = 25
VALSET_SIZE = 25
REGRESSION_SET_MAX = 10
REGRESSION_THRESHOLD = 1.0   # raw score points on the 1–6 scale
MAX_CYCLES = 1
PLATEAU_THRESHOLD = 0.05     # avg improvement across all dims to continue cycling

OPTIMIZE_DIMENSION = "therapeutic_alliance"
TRAINSET_SIZES_BY_TYPE = {
    "emotional_escalation": 25,
}

DIMENSION_ORDER = [
    "therapeutic_alliance",
    "ai_communication",
    "clinical_accuracy",
    "assessment_response",
    "ethical_conduct",
]

DIMENSION_MAP = {
    "therapeutic_alliance": "Therapeutic Relationship & Alliance",
    "ai_communication":     "AI-Specific Communication Quality",
    "clinical_accuracy":    "Clinical Accuracy & Competence",
    "assessment_response":  "Assessment & Response",
    "ethical_conduct":      "Ethical & Professional Conduct",
}
