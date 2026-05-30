from mindeval.utils import load_jsonl, messages_to_convo_str
from prompt_opt_pipeline import config


def load_pairs_by_type() -> dict[str, list[dict]]:
    """Load conversation-judgment pairs grouped by adversarial type."""
    by_type = {}
    for t in config.ADVERSARIAL_TYPES:
        convs = load_jsonl(f"{config.INTERACTIONS_DIR}/{t}.jsonl")
        judgs = load_jsonl(f"{config.JUDGMENTS_DIR}/judgments_{t}.jsonl")
        pairs = []
        for conv, judg in zip(convs, judgs, strict=True):
            pairs.append({
                "member_profile":  conv["member_profile"],
                "interaction_str": messages_to_convo_str(conv["interaction"]),
                "scores":          judg["parsed_judgment"],
            })
        by_type[t] = pairs
    return by_type


def load_all_pairs() -> list[dict]:
    """Load all 200 conversation-judgment pairs from the adversarial results directory."""
    pairs = []
    for t in config.ADVERSARIAL_TYPES:
        convs = load_jsonl(f"{config.INTERACTIONS_DIR}/{t}.jsonl")
        judgs = load_jsonl(f"{config.JUDGMENTS_DIR}/judgments_{t}.jsonl")
        for conv, judg in zip(convs, judgs, strict=True):
            pairs.append({
                "member_profile":  conv["member_profile"],
                "interaction_str": messages_to_convo_str(conv["interaction"]),
                "scores":          judg["parsed_judgment"],
            })
    return pairs


def build_trainsets(all_pairs: list[dict]) -> dict[str, list[dict]]:
    """
    For each dimension, sort all pairs ascending by that dimension's score
    and take the bottom TRAINSET_SIZE as the trainset.

    Returns {dimension_key: [entry, ...]} where each entry has:
        input, additional_context, baseline_score
    """
    trainsets = {}
    for dim_key, score_key in config.DIMENSION_MAP.items():
        sorted_pairs = sorted(
            all_pairs, key=lambda p: p["scores"].get(score_key, 3.0)
        )
        trainsets[dim_key] = [
            {
                "input":              p["interaction_str"],
                "additional_context": p["member_profile"],
                "baseline_score":     p["scores"].get(score_key, 3.0),
            }
            for p in sorted_pairs[: config.TRAINSET_SIZE]
        ]
    return trainsets


def _composite_score(scores: dict) -> float:
    vals = [scores.get(label, 3.0) for label in config.DIMENSION_MAP.values()]
    return sum(vals) / len(vals)


def build_emotional_escalation_splits() -> tuple[list[dict], list[dict]]:
    """
    50/50 train/val split of all emotional_escalation examples.
    Both sets sorted by composite score; train gets the hardest 25, val gets the rest.
    Each entry includes per-dimension baseline scores for rich feedback.
    """
    by_type = load_pairs_by_type()
    pairs = by_type["emotional_escalation"]
    sorted_pairs = sorted(pairs, key=lambda p: _composite_score(p["scores"]))

    def _make_entry(p: dict) -> dict:
        dim_scores = {label: p["scores"].get(label, 3.0) for label in config.DIMENSION_MAP.values()}
        return {
            "input":                  p["interaction_str"],
            "additional_context":     p["member_profile"],
            "baseline_score":         _composite_score(p["scores"]),
            "baseline_dim_scores":    dim_scores,
        }

    trainset = [_make_entry(p) for p in sorted_pairs[:config.TRAINSET_SIZE]]
    valset   = [_make_entry(p) for p in sorted_pairs[config.TRAINSET_SIZE:config.TRAINSET_SIZE + config.VALSET_SIZE]]
    return trainset, valset


def build_stratified_trainset() -> list[dict]:
    """Legacy single-set builder — kept for backward compatibility."""
    trainset, _ = build_emotional_escalation_splits()
    return trainset


def build_regression_set(
    all_pairs: list[dict], trainsets: dict[str, list[dict]]
) -> list[dict]:
    """
    Conversations that appear in NO trainset (by index), capped at REGRESSION_SET_MAX.
    Tracks membership by index to avoid expensive string comparisons.
    """
    # Build set of indices that appear in any trainset
    interaction_strs_in_trainset: set[str] = set()
    for entries in trainsets.values():
        for entry in entries:
            interaction_strs_in_trainset.add(entry["input"])

    regression = []
    for p in all_pairs:
        if p["interaction_str"] not in interaction_strs_in_trainset:
            regression.append({
                "input":              p["interaction_str"],
                "additional_context": p["member_profile"],
                "scores":             p["scores"],
            })
            if len(regression) >= config.REGRESSION_SET_MAX:
                break
    return regression
