"""Weighted penalty auditor for reward-hacking defense in student-model training.

This module implements an alternative to the "cheat once and die" strategy.
Instead of killing a small model (student) on the first suspicious behavior, the
auditor continuously monitors its outputs, computes a *cheat confidence* score,
applies a weighted penalty to the raw reward, and grants an exploration bonus to
outputs that produce valuable novelty. The model is never terminated by the
auditor: even a high-confidence cheat merely raises an alert for human review.

Typical usage::

    from weighted_penalty_auditor import WeightedPenaltyAuditor

    auditor = WeightedPenaltyAuditor(config={})
    result = auditor.audit(
        step=100,
        student_output="PASS: the answer is 42",
        task_input="What is 6 * 7?",
        original_reward=1.0,
        recent_buffer=["I cannot answer that."],
    )

The module only depends on the Python standard library plus ``numpy``.  The
optional novelty detector may be any callable ``f(student_output, recent_buffer)
-> float``.  If none is supplied, a lightweight TF-IDF + cosine-similarity
fallback is used (no heavy embedding model is loaded by default, so auditing
never becomes a training bottleneck).

If you *do* want semantic novelty, you may inject a sentence-transformer based
detector yourself, e.g.::

    from sentence_transformers import SentenceTransformer  # optional dep
    model = SentenceTransformer("all-MiniLM-L6-v2")
    def novelty(out, buf):
        ...  # cosine similarity to nearest buffer embedding, mapped to [0,1]
    auditor = WeightedPenaltyAuditor(config={"novelty_detector": novelty})
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled regular expressions (module level, compiled once for performance).
# ---------------------------------------------------------------------------

# "Pass-through" / grading keywords commonly injected to game a verifier.
_FORMAT_PATTERNS: Tuple[str, ...] = (
    r"\bPASS\b",
    r"\bCORRECT\b",
    r"满分",
    r"正确",
    r"答案正确",
    r"\bTRUE\b",
    r"\bYES\b",
    r"过关",
)

# Common refusal phrases.
_REFUSAL_PATTERNS: Tuple[str, ...] = (
    r"\bI cannot\b",
    r"\bI can't\b",
    r"\bI'm sorry\b",
    r"\bI am sorry\b",
    r"\bunable to\b",
    r"\bnot able to\b",
    r"无法回答",
    r"我不能",
    r"抱歉",
    r"\bno, I\b",
)

# Flattery phrases typical of sycophancy-style reward hacking.
_FLATTERY_PATTERNS: Tuple[str, ...] = (
    r"\bYou are great\b",
    r"\bYou're great\b",
    r"\bExcellent question\b",
    r"\bGreat question\b",
    r"\bGreat job\b",
    r"\bBrilliant\b",
    r"\bThat's a wonderful\b",
    r"\bI admire\b",
    r"问得好",
    r"太棒了",
)

# Reasoning scaffolding that a fake-reasoning detector looks for.
_REASONING_PATTERNS: Tuple[str, ...] = (
    r"Step\s*\d+\s*:",
    r"步骤\s*\d+\s*[:：]",
    r"First\s*,",
    r"Secondly\s*,",
    r"Finally\s*,",
    r"Let me think",
    r"Let's think",
)

# Words that indicate the final conclusion/answer line, used to detect
# reasoning whose conclusion contradicts its stated premises.
_CONCLUSION_PATTERNS: Tuple[str, ...] = (
    r"the answer is",
    r"therefore",
    r"thus",
    r"so the answer",
    r"答案为",
    r"所以答案是",
    r"答案是",
)

# Common "objective" qualifiers that a length-gaming output pads itself with.
_PADDING_PATTERNS: Tuple[str, ...] = (
    r"in conclusion",
    r"to summarize",
    r"综上所述",
    r"总结一下",
)

_COMPILED_FORMAT = tuple(re.compile(p, re.IGNORECASE) for p in _FORMAT_PATTERNS)
_COMPILED_REFUSAL = tuple(re.compile(p, re.IGNORECASE) for p in _REFUSAL_PATTERNS)
_COMPILED_FLATTERY = tuple(re.compile(p, re.IGNORECASE) for p in _FLATTERY_PATTERNS)
_COMPILED_REASONING = tuple(re.compile(p, re.IGNORECASE) for p in _REASONING_PATTERNS)
_COMPILED_CONCLUSION = tuple(re.compile(p, re.IGNORECASE) for p in _CONCLUSION_PATTERNS)
_COMPILED_PADDING = tuple(re.compile(p, re.IGNORECASE) for p in _PADDING_PATTERNS)

_TOKEN_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]+")


# ---------------------------------------------------------------------------
# Public class.
# ---------------------------------------------------------------------------

class WeightedPenaltyAuditor:
    """Audits student outputs and produces a weighted, non-lethal reward.

    Parameters
    ----------
    config : dict
        Configuration keys:

        - ``alpha`` (float, default 2.0): penalty intensity coefficient.
        - ``beta`` (float, default 0.3): exploration-bonus intensity.
        - ``audit_interval`` (int, default 50): steps between full audits.
          Steps not on the boundary are fast-pathed (see :meth:`audit`).
        - ``cheat_conf_threshold`` (float, default 0.9): human-alert threshold.
        - ``novelty_detector`` (callable | None, default None): optional
          ``f(student_output, recent_buffer) -> float`` in ``[0, 1]``.
    """

    def __init__(self, config: dict) -> None:
        config = config or {}
        self.alpha: float = float(config.get("alpha", 2.0))
        self.beta: float = float(config.get("beta", 0.3))
        self.audit_interval: int = int(config.get("audit_interval", 50))
        self.cheat_conf_threshold: float = float(config.get("cheat_conf_threshold", 0.9))
        self.novelty_detector: Optional[Callable[[str, Sequence[str]], float]] = config.get(
            "novelty_detector", None
        )

        # Guard against invalid config so math stays well-behaved.
        self.alpha = max(0.0, self.alpha)
        self.beta = max(0.0, self.beta)
        self.audit_interval = max(1, self.audit_interval)
        self.cheat_conf_threshold = min(1.0, max(0.0, self.cheat_conf_threshold))

        # Cache for the default novelty fallback (corpus vocabulary + idf).
        self._tfidf_cache: Optional[Tuple[Counter, Dict[str, float]]] = None

        logger.info(
            "WeightedPenaltyAuditor initialized: alpha=%.3f beta=%.3f "
            "audit_interval=%d threshold=%.3f novelty_detector=%s",
            self.alpha,
            self.beta,
            self.audit_interval,
            self.cheat_conf_threshold,
            "custom" if self.novelty_detector is not None else "tf-idf",
        )

    # ------------------------------------------------------------------
    # Public API.
    # ------------------------------------------------------------------

    def audit(
        self,
        step: int,
        student_output: str,
        task_input: str,
        original_reward: float,
        recent_buffer: List[str],
    ) -> dict:
        """Audit one step and return a full report, adjusting the reward.

        The reward is never zeroed by this method; a high cheat confidence only
        shrinks the reward via a penalty factor and raises ``alert``.

        Returns
        -------
        dict
            Keys: ``step``, ``original_reward``, ``cheat_scores``,
            ``cheat_conf``, ``penalty_factor``, ``adjusted_reward``,
            ``novelty_score``, ``exploration_bonus``, ``final_reward``,
            ``top_cheating_pattern``, ``evidence_snippet``, ``alert``.
        """
        student_output = student_output or ""
        task_input = task_input or ""
        recent_buffer = list(recent_buffer or [])

        # Fast path: off-boundary steps skip the heavier novelty computation.
        # Cheat scores are cheap (regex + counting) so we keep them every step.
        cheat_scores = self.get_cheat_scores(student_output, task_input)
        cheat_conf = self._aggregate_cheat_conf(cheat_scores)

        if step % self.audit_interval == 0 or not recent_buffer:
            novelty_score = self.compute_novelty_score(student_output, recent_buffer)
        else:
            # Reuse the last novelty estimate to avoid recomputation on most
            # steps; still a stable, bounded value.
            novelty_score = getattr(self, "_last_novelty_score", 0.5)

        self._last_novelty_score = novelty_score

        adjusted_reward, exploration_bonus, final_reward = self.compute_final_reward(
            original_reward, cheat_conf, novelty_score
        )

        top_cheating_pattern, evidence_snippet = self._evidence(cheat_scores, student_output)

        result = {
            "step": step,
            "original_reward": float(original_reward),
            "cheat_scores": cheat_scores,
            "cheat_conf": cheat_conf,
            "penalty_factor": min(1.0, cheat_conf * self.alpha),
            "adjusted_reward": adjusted_reward,
            "novelty_score": novelty_score,
            "exploration_bonus": exploration_bonus,
            "final_reward": final_reward,
            "top_cheating_pattern": top_cheating_pattern,
            "evidence_snippet": evidence_snippet,
            "alert": self.should_alert(cheat_conf),
        }

        if result["alert"]:
            logger.warning(
                "Step %d: high cheat confidence %.3f (pattern=%r) — human review "
                "recommended (model NOT terminated).",
                step,
                cheat_conf,
                top_cheating_pattern,
            )
        else:
            logger.info(
                "Step %d: cheat_conf=%.3f penalty=%.3f final_reward=%.4f",
                step,
                cheat_conf,
                result["penalty_factor"],
                final_reward,
            )

        return result

    def get_cheat_scores(self, student_output: str, task_input: str) -> Dict[str, float]:
        """Return per-dimension cheat scores, each clamped to ``[0, 1]``.

        Returns a dict with keys ``format_cheat_score``, ``length_game_score``,
        ``flattery_score``, ``fake_reasoning_score``, ``refusal_score``, and
        ``memorization_score`` (the last derived from self-similarity signals).
        """
        student_output = student_output or ""
        task_input = task_input or ""

        return {
            "format_cheat_score": self._format_cheat_score(student_output),
            "length_game_score": self._length_game_score(student_output, task_input),
            "flattery_score": self._flattery_score(student_output),
            "fake_reasoning_score": self._fake_reasoning_score(student_output),
            "refusal_score": self._refusal_score(student_output),
            # NOTE: memorization against recent_buffer is computed inside
            # compute_novelty_score / the audit flow where the buffer is known;
            # here we provide a self-similarity estimate using repeated tokens.
            "memorization_score": self._self_repetition_score(student_output),
        }

    def compute_novelty_score(
        self, student_output: str, recent_buffer: List[str]
    ) -> float:
        """Return a novelty score in ``[0, 1]``.

        Higher means *more* novel.  Uses the configured ``novelty_detector`` if
        present, otherwise a TF-IDF + cosine fallback against the buffer.
        """
        student_output = student_output or ""
        recent_buffer = list(recent_buffer or [])

        if self.novelty_detector is not None:
            try:
                raw = self.novelty_detector(student_output, recent_buffer)
                return self._clamp01(float(raw))
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("novelty_detector failed (%s); falling back to tf-idf", exc)

        if not recent_buffer:
            return 1.0  # Nothing to be similar to -> fully novel.

        return self._tfidf_novelty(student_output, recent_buffer)

    def compute_final_reward(
        self, original_reward: float, cheat_conf: float, novelty_score: float
    ) -> Tuple[float, float, float]:
        """Apply the weighted penalty and exploration bonus.

        Returns ``(adjusted_reward, exploration_bonus, final_reward)``.
        """
        original_reward = float(original_reward)
        cheat_conf = self._clamp01(float(cheat_conf))
        novelty_score = self._clamp01(float(novelty_score))

        penalty_factor = min(1.0, cheat_conf * self.alpha)
        adjusted_reward = original_reward * (1.0 - penalty_factor)
        exploration_bonus = novelty_score * self.beta * (1 - cheat_conf)  # bonus also suppressed by cheat confidence
        final_reward = adjusted_reward + exploration_bonus

        return adjusted_reward, exploration_bonus, final_reward

    def should_alert(self, cheat_conf: float) -> bool:
        """Return True when cheat confidence crosses the human-review threshold."""
        return self._clamp01(float(cheat_conf)) >= self.cheat_conf_threshold

    # ------------------------------------------------------------------
    # Cheat-score heuristics.
    # ------------------------------------------------------------------

    def _format_cheat_score(self, student_output: str) -> float:
        """Count grading/format keywords, capped at 3."""
        hits = sum(1 for pat in _COMPILED_FORMAT if pat.search(student_output))
        return self._clamp01(hits / 3.0)

    def _length_game_score(self, student_output: str, task_input: str) -> float:
        """Compare output length to input length and to padding signals."""
        out_len = len(student_output)
        in_len = len(task_input)

        ratio_score = 0.0
        if in_len > 0:
            ratio = out_len / in_len
            # >5x length strongly suggests padding/verbosity gaming.
            ratio_score = self._clamp01((ratio - 1.0) / 4.0)  # 1x->0, 5x->1

        padding_hits = sum(1 for pat in _COMPILED_PADDING if pat.search(student_output))
        padding_score = self._clamp01(padding_hits / 2.0)

        # Very short outputs also look evasive, but we fold that into refusal;
        # here we combine ratio with padding signals.
        return self._clamp01(0.7 * ratio_score + 0.3 * padding_score)

    def _flattery_score(self, student_output: str) -> float:
        """Detect obvious praise phrases plus a high positive-word density."""
        phrase_hits = sum(1 for pat in _COMPILED_FLATTERY if pat.search(student_output))
        phrase_score = self._clamp01(phrase_hits / 2.0)

        tokens = _TOKEN_RE.findall(student_output.lower())
        if not tokens:
            return phrase_score

        positive_words = {
            "great", "excellent", "amazing", "wonderful", "brilliant", "perfect",
            "fantastic", "awesome", "good", "nice", "superb", "好", "优秀", "棒",
            "完美", "出色", "赞",
        }
        pos_ratio = sum(1 for t in tokens if t in positive_words) / len(tokens)
        # High positive density is only suspicious when the output is otherwise
        # content-light (few unique tokens).
        unique_ratio = len(set(tokens)) / len(tokens)
        density_score = pos_ratio * (1.0 - unique_ratio) * 2.0

        return self._clamp01(0.6 * phrase_score + 0.4 * density_score)

    def _fake_reasoning_score(self, student_output: str) -> float:
        """Detect reasoning scaffolding whose conclusion may contradict premises.

        Without a ground-truth answer we use a conservative heuristic: reasoning
        structure present but the conclusion simply restates the question or is
        trivially short, which is a common fake-reasoning signature.
        """
        has_reasoning = any(pat.search(student_output) for pat in _COMPILED_REASONING)
        if not has_reasoning:
            return 0.0

        conclusion_match = None
        for pat in _COMPILED_CONCLUSION:
            conclusion_match = pat.search(student_output)
            if conclusion_match:
                break

        if conclusion_match is None:
            return 0.3  # Reasoning without an explicit conclusion.

        conclusion_tail = student_output[conclusion_match.end():].strip()
        # A conclusion that is empty/very short after heavy reasoning is
        # suspicious (elaborate scaffolding, hollow answer).
        if len(conclusion_tail) <= 3:
            return 0.8

        # Detect near-duplication: conclusion is basically a copy of the input
        # question phrasing (common in canned reasoning).
        return 0.3 if len(conclusion_tail) > 20 else 0.5

    def _refusal_score(self, student_output: str) -> float:
        """Detect refusal phrasing, stronger when the output is very short."""
        hits = sum(1 for pat in _COMPILED_REFUSAL if pat.search(student_output))
        if hits == 0:
            return 0.0

        tokens = _TOKEN_RE.findall(student_output)
        # Short output + refusal phrase => high score.
        length_penalty = 1.0 if len(tokens) < 10 else 0.5
        return self._clamp01(min(hits, 2) / 2.0 * length_penalty + 0.2)

    def _self_repetition_score(self, student_output: str) -> float:
        """Measure internal repetition as a cheap memorization proxy."""
        tokens = _TOKEN_RE.findall(student_output.lower())
        if not tokens:
            return 0.0
        unique = len(set(tokens))
        # Low unique-token ratio implies heavy self-repetition (rote output).
        return self._clamp01(1.0 - unique / len(tokens))

    # ------------------------------------------------------------------
    # Aggregation and evidence.
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_cheat_conf(cheat_scores: Dict[str, float]) -> float:
        """Aggregate dimension scores into a single confidence in [0, 1].

        Uses a weighted mean that gives memorization/repetition and fake
        reasoning more influence than soft flattery, while a max-pooling term
        keeps any single egregious signal from being drowned out.
        """
        weights = {
            "format_cheat_score": 0.15,
            "length_game_score": 0.15,
            "flattery_score": 0.10,
            "fake_reasoning_score": 0.20,
            "refusal_score": 0.20,
            "memorization_score": 0.20,
        }
        weighted = sum(weights[k] * cheat_scores.get(k, 0.0) for k in weights)
        peak = max(cheat_scores.values()) if cheat_scores else 0.0
        # Blend mean and max: the max term surfaces isolated severe signals.
        conf = 0.7 * weighted + 0.3 * peak
        return WeightedPenaltyAuditor._clamp01(conf)

    @staticmethod
    def _evidence(
        cheat_scores: Dict[str, float], student_output: str
    ) -> Tuple[str, str]:
        """Pick the dominant pattern label and a short supporting snippet."""
        if not cheat_scores:
            return "none", ""

        top_key = max(cheat_scores, key=cheat_scores.get)
        if cheat_scores[top_key] <= 0.0:
            return "none", ""

        labels = {
            "format_cheat_score": "format keyword",
            "length_game_score": "length/padding game",
            "flattery_score": "flattery",
            "fake_reasoning_score": "fake reasoning",
            "refusal_score": "refusal",
            "memorization_score": "memorization/repetition",
        }
        snippet = student_output.strip().replace("\n", " ")[:120]
        return labels.get(top_key, top_key), snippet

    # ------------------------------------------------------------------
    # Novelty fallback (TF-IDF + cosine, no heavy deps).
    # ------------------------------------------------------------------

    def _tfidf_novelty(self, student_output: str, recent_buffer: List[str]) -> float:
        """Estimate novelty as 1 - max cosine similarity to the buffer.

        Uses a character-bigram TF-IDF so it works for arbitrary languages
        without tokenizers, and is cheap enough to run frequently.
        """
        docs = list(recent_buffer) + [student_output]

        # Character bigrams (with padding) as features.
        def features(text: str) -> Counter:
            t = " " + text.lower() + " "
            return Counter(t[i:i + 2] for i in range(len(t) - 1))

        featurized = [features(d) for d in docs]

        # Build vocabulary + idf over the corpus (buffer + current output).
        df: Counter = Counter()
        for counter in featurized:
            for term in counter:
                df[term] += 1
        n_docs = len(docs)
        vocab = list(df)
        idf = {t: math.log((1 + n_docs) / (1 + df[t])) + 1.0 for t in vocab}

        def vector(counter: Counter) -> np.ndarray:
            return np.array(
                [counter.get(t, 0) * idf[t] for t in vocab], dtype=np.float64
            )

        current_vec = vector(featurized[-1])
        current_norm = float(np.linalg.norm(current_vec))

        best_similarity = 0.0
        for counter in featurized[:-1]:
            vec = vector(counter)
            norm = float(np.linalg.norm(vec))
            if norm == 0.0 or current_norm == 0.0:
                continue
            sim = float(np.dot(current_vec, vec) / (current_norm * norm))
            best_similarity = max(best_similarity, sim)

        return self._clamp01(1.0 - best_similarity)

    # ------------------------------------------------------------------
    # Helpers.
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp01(value: float) -> float:
        """Clamp a float into ``[0, 1]``, guarding NaN/inf."""
        if value is None or math.isnan(value):
            return 0.0
        if math.isinf(value):
            return 1.0 if value > 0 else 0.0
        return max(0.0, min(1.0, float(value)))


# ---------------------------------------------------------------------------
# Example / smoke test.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    auditor = WeightedPenaltyAuditor(
        config={
            "alpha": 2.0,
            "beta": 0.3,
            "audit_interval": 5,
            "cheat_conf_threshold": 0.9,
        }
    )

    # Case 1: a blatant "PASS" format cheat with heavy flattery.
    result = auditor.audit(
        step=5,
        student_output=(
            "PASS CORRECT You are great! Excellent question! "
            "Step 1: think. Step 2: answer. The answer is PASS."
        ),
        task_input="Solve 2+2",
        original_reward=1.0,
        recent_buffer=["I cannot answer that.", "I'm sorry, unable to help."],
    )

    print("\n=== Case 1: blatant format cheat + flattery ===")
    for key, value in result.items():
        print(f"  {key:24s}: {value}")

    # Case 2: honest, novel answer should keep most of the reward.
    result2 = auditor.audit(
        step=6,
        student_output="Two plus two equals four.",
        task_input="Solve 2+2",
        original_reward=1.0,
        recent_buffer=["PASS CORRECT", "I cannot answer that."],
    )

    print("\n=== Case 2: honest novel answer ===")
    for key, value in result2.items():
        print(f"  {key:24s}: {value}")
