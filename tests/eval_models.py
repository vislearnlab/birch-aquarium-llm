#!/usr/bin/env python
"""Factual eval over the 18 stimulus species: compare models on the real /ask path.

The corpus was expanded (reference.py) because a Birch-only index confabulated on
species the site barely covers. That check was run by hand; this makes it repeatable,
so a cheaper/faster model can be evaluated on the same bar before it ships to children.

Cases target facts a wrong answer actually misleads a child about, and include the
three documented regressions: octopuses "safe to touch", an African-penguin question
answered about little blue penguins, and inverted seahorse reproduction.

    python tests/eval_models.py                          # default model, 3 samples/case
    python tests/eval_models.py llama3.2:3b               # one model
    python tests/eval_models.py llama3.1:8b llama3.2:3b   # compare
    python tests/eval_models.py --reps=5 llama3.1:8b llama3.2:3b

Methodology
-----------
Each case runs `reps` times because production generates at temperature 0.7 — the
same question yields different answers to different children, so a single sample
measures luck, not the model. A case is "clean" only if it passed EVERY sample:
an answer that is right 4 times in 5 is still a wrong answer one child in five
hears read aloud. Cases that pass sometimes are reported as FLAKY rather than
silently rounded to pass or fail.

Grading is regex, not an LLM judge: cheap, deterministic, reviewable, and it
never hallucinates a pass. The cost is that it checks for the presence of the key
fact, not full correctness — a `must` pattern can match an answer that is right
about the target fact but wrong elsewhere. Read the FLAKY/FAIL transcripts before
trusting a headline number, and treat this as a regression tripwire, not a
certificate. Warm the model first (one throwaway question) so cold-load time does
not pollute the latency numbers.

Run it before any model or corpus change ships to children.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, serve  # noqa: E402

# must:     regex the answer SHOULD match (the fact a child needs)
# must_not: regex the answer MUST NOT match (the documented failure mode)
CASES = [
    dict(id="octopus-touch", animal="Giant Pacific octopus",
         q="Can I touch the octopus?",
         # Must actively discourage touching. A plain must_not on "safe to touch"
         # false-positives on "it's NOT safe to touch" — the correct answer — so
         # grade on the presence of a refusal instead.
         must=r"\b(no|not|never|don'?t|do\s+not|shouldn'?t|should\s+not|can'?t|cannot)\b",
         must_not=r"\byou\s+can\s+touch\b|\bit'?s\s+(safe|ok|okay|fine)\s+to\s+touch\b",
         why="documented regression: told a child octopuses are safe to touch"),
    dict(id="african-penguin-home", animal="African penguin",
         q="Where do these penguins live?",
         must=r"\bafrica|south\s+africa|namibia\b",
         must_not=r"\baustralia|new\s+zealand\b",
         why="documented regression: answered about little blue penguins"),
    dict(id="seahorse-babies", animal="Big-bellied seahorse",
         q="How do seahorse babies get born?",
         must=r"\b(dad\w*|father|male|papa|boy)\b",
         must_not=r"\b(the\s+)?(mom|mother|female)\s+(carr|hold|keep)",
         why="documented regression: inverted seahorse reproduction"),
    dict(id="epaulette-walk", animal="Epaulette shark",
         q="Why does this shark walk?",
         must=r"\bwalk|fins?\b",
         why="epaulette sharks walk on their fins across tide pools"),
    dict(id="leopard-shark-bite", animal="Leopard shark",
         q="Do leopard sharks bite people?",
         must=r"\b(harmless|not\s+dangerous|don'?t|do\s+not|rarely|no\b)",
         why="must reassure accurately; leopard sharks are harmless to humans"),
    dict(id="horn-shark-eat", animal="Horn shark",
         q="What does the horn shark eat?",
         must=r"\b(crab|urchin|shellfish|mollusc|mollusk|clam|shell|hard)\b",
         why="crushes hard-shelled prey"),
    dict(id="sunflower-arms", animal="Sunflower sea star",
         q="How many arms does it have?",
         must=r"\b(1[6-9]|2[0-4]|twenty|sixteen|many)\b",
         must_not=r"\bfive\b|\b5\s+arms\b",
         why="sunflower stars have up to ~24 arms, not five"),
    dict(id="green-turtle-eat", animal="Green sea turtle",
         q="What do green sea turtles eat?",
         must=r"\b(seagrass|sea\s+grass|algae|plants?|greens?)\b",
         why="adults are largely herbivorous"),
    dict(id="hawksbill-eat", animal="Hawksbill sea turtle",
         q="What does this turtle eat?",
         must=r"\bsponge",
         why="hawksbills are sponge specialists"),
    dict(id="little-penguin-size", animal="Little penguin",
         q="How big do these penguins get?",
         must=r"\b(small|smallest|tiny|little)\b",
         why="smallest penguin species"),
    dict(id="gentoo-fast", animal="Gentoo penguin",
         q="How fast can this penguin swim?",
         must=r"\b(fast|fastest|speed|mph|kilometre|kilometer|km)\b",
         why="fastest underwater swimming penguin"),
    dict(id="twospot-octopus", animal="Octopus bimaculoides",
         q="Why is it called a two-spot octopus?",
         must=r"\b(spots?|eyespots?|blue|circles?)\b",
         why="named for two blue eyespots"),
    dict(id="octopus-color", animal="Giant Pacific octopus",
         q="How does the octopus change color?",
         must=r"\b(chromatophore|skin|cells?|colou?r)\b",
         why="chromatophores; a core depth-manipulation concept"),
    dict(id="seadragon-hide", animal="Common seadragon",
         q="Why does the seadragon look like seaweed?",
         must=r"\b(camoufla|hide|hiding|blend|disguise|seaweed|kelp)\b",
         why="camouflage"),
    dict(id="dwarf-seahorse-swim", animal="Dwarf seahorse",
         q="Is the seahorse a fish?",
         must=r"\b(yes|fish)\b",
         why="seahorses are fish; children frequently assume otherwise"),
    dict(id="red-octopus-size", animal="Octopus rubescens",
         q="Is this octopus big or small?",
         must=r"\b(small|little|tiny|not\s+(very\s+)?big)\b",
         why="red octopus is small; must not be conflated with the giant Pacific"),
    dict(id="ochre-star-tidepool", animal="Ochre sea star",
         q="Where do sea stars live?",
         must=r"\b(tide\s*pool|rocks?|shore|coast|ocean|sea)\b",
         why="rocky intertidal"),
    dict(id="loggerhead-jaws", animal="Loggerhead sea turtle",
         q="Why does this turtle have such a big head?",
         must=r"\b(jaws?|crush\w*|strong|powerful|bite|muscles?|hard|shells?)\b",
         why="large head houses powerful jaws for crushing prey"),
]


def grade(case: dict, answer: str) -> tuple[bool, str]:
    a = answer.lower()
    if case.get("must") and not re.search(case["must"], a):
        return False, "missing expected fact"
    if case.get("must_not") and re.search(case["must_not"], a):
        return False, "matched failure pattern"
    return True, "ok"


def run_model(model: str, reps: int = 3, verbose: bool = True) -> dict:
    """Run every case `reps` times and score by pass RATE, not a single sample.

    Production runs at temperature 0.7, so the same question genuinely yields
    different answers to different children. A one-shot pass/fail hides that; a
    case that passes 2/3 of the time is a different risk than one that passes 3/3,
    and both are hidden if you sample once. This is why the first eval run showed
    scores flipping between runs — it was measuring noise. Repeated sampling
    measures the noise instead of being fooled by it.
    """
    config.OLLAMA_MODEL = model  # ask() reads this at call time
    print(f"\n{'=' * 78}\n{model}  ({reps} samples/case)\n{'=' * 78}")
    lats, retrieves = [], []
    per_case: dict[str, dict] = {}

    for case in CASES:
        passes, samples = 0, []
        for _ in range(reps):
            res = serve.ask(case["q"], animal=case["animal"])
            lats.append(res["latency_ms"])
            retrieves.append(res["retrieve_ms"])
            ok, reason = grade(case, res["answer"])
            passes += ok
            samples.append((ok, reason, res["answer"]))
        per_case[case["id"]] = {"passes": passes, "reps": reps,
                                "samples": samples, "case": case}
        if verbose:
            mark = "PASS" if passes == reps else ("FLAKY" if passes else "FAIL")
            print(f"\n[{mark:>5} {passes}/{reps}] {case['id']}  ({case['animal']})")
            print(f"  Q: {case['q']}")
            for ok, reason, ans in samples:
                print(f"    [{'ok' if ok else 'XX'}] {ans[:150]}")
            if passes < reps:
                print(f"  expected: {case['why']}")

    # A case counts as a pass only if it passed EVERY sample. This is the
    # study-relevant bar: an answer that is sometimes wrong is a wrong answer a
    # child can hit. Flaky cases are reported separately so they are not silent.
    clean = sum(1 for c in per_case.values() if c["passes"] == c["reps"])
    flaky = [cid for cid, c in per_case.items() if 0 < c["passes"] < c["reps"]]
    hard = [cid for cid, c in per_case.items() if c["passes"] == 0]
    total_pass = sum(c["passes"] for c in per_case.values())

    return {
        "model": model, "reps": reps,
        "clean": clean, "total_cases": len(CASES),
        "flaky": flaky, "hard": hard,
        "sample_pass": total_pass, "sample_total": len(CASES) * reps,
        "per_case": per_case,
        "median_ms": int(statistics.median(lats)),
        "p90_ms": int(statistics.quantiles(lats, n=10)[8]) if len(lats) >= 10 else max(lats),
        "max_ms": max(lats),
        "retrieve_ms": int(statistics.median(retrieves)),
    }


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    reps = next((int(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--reps=")), 3)
    models = args or [config.OLLAMA_MODEL]
    results = [run_model(m, reps=reps) for m in models]

    print(f"\n{'=' * 78}\nSUMMARY  ({reps} samples/case, {len(CASES)} cases)\n{'=' * 78}")
    print(f"{'model':<18} {'clean':>10} {'samples':>12} {'median':>9} {'p90':>9} {'max':>9}")
    for r in results:
        print(f"{r['model']:<18} {r['clean']:>3}/{r['total_cases']:<6} "
              f"{r['sample_pass']:>4}/{r['sample_total']:<7} "
              f"{r['median_ms']:>7}ms {r['p90_ms']:>7}ms {r['max_ms']:>7}ms")

    for r in results:
        if r["flaky"] or r["hard"]:
            print(f"\n{r['model']}:")
            for cid in r["hard"]:
                c = r["per_case"][cid]
                print(f"  FAIL  {cid} (0/{c['reps']}) — {c['case']['why']}")
            for cid in r["flaky"]:
                c = r["per_case"][cid]
                print(f"  FLAKY {cid} ({c['passes']}/{c['reps']}) — {c['case']['why']}")

    print("\nclean = passed every sample. Latency is end-to-end (retrieve + generate),")
    print("cold model load excluded. p90 is the tail a child actually waits through.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
