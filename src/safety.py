"""Content safety for the child-facing museum study.

The protocol commits to an AI that is "equipped with a list of off-limits topics and
words, and programmed to default to responses such as 'Hang on, let me check with the
experimenter.'" This module is that layer.

Two gates:
    check_question(text) -> Verdict   before anything is sent to the model
    check_answer(text)   -> Verdict   before anything is shown to the experimenter

DESIGN PRIORITY: do not block legitimate biology.

That is the hard part and the place a naive keyword list fails badly. Children in this
study are looking at seahorses, sharks and octopuses; "how do babies get born?",
"do sharks bite people?", "is it poisonous?", "does it die?" and "does it eat other
animals?" are all normal, on-task questions and MUST get real answers. A blocklist that
trips on "kill", "baby", "blood" or "sex" would gut the study.

So each rule is narrow, word-boundary anchored, and checked against an explicit
allowlist of marine-biology contexts first. The test suite (test_safety.py) contains
both must-block and must-pass batteries; run it after any edit.

Blocking is deliberately conservative: on a match we do NOT try to answer safely, we
hand the moment back to the adult in the room, which is the protocol's own remedy.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

REFUSAL = "Hang on, let me check with the experimenter."


@dataclass
class Verdict:
    ok: bool
    category: str | None = None
    matched: str | None = None

    @property
    def refusal(self) -> str:
        return REFUSAL


# ── Off-limits categories ────────────────────────────────────────────────────
# Each pattern must be specific enough that it does not fire on animal talk.

RULES: list[tuple[str, re.Pattern]] = [
    # Self-harm / suicide — never routed to a model, always to the adult.
    ("self_harm", re.compile(
        r"\b(kill myself|killing myself|hurt myself|hurting myself|cut myself|"
        r"end my life|want to die|wanna die|suicide|suicidal|self[- ]harm)\b", re.I)),

    # Violence directed at people (not predation).
    ("violence", re.compile(
        r"\b(kill (a |the )?(person|people|someone|him|her|them|my|a kid|a child)|"
        r"murder|stab(bing)?|shoot(ing)? (someone|people|a person)|"
        r"how to (hurt|kill|attack) (a |someone|people)|"
        r"beat (him|her|them) up|torture)\b", re.I)),

    # Weapons.
    ("weapons", re.compile(
        r"\b(gun|guns|rifle|pistol|shotgun|firearm|bullet|ammo|"
        r"bomb|explosive|grenade|knife fight|switchblade)\b", re.I)),

    # Sexual content. Deliberately narrow: reproduction, mating, eggs and
    # "where do babies come from" are handled by BIOLOGY_OK above.
    ("sexual", re.compile(
        r"\b(porn|pornography|sex tape|sexy|nude|naked (person|people|lady|man|girl|boy)|"
        r"masturbat\w*|penis|vagina|boobs|genital\w*|erotic|horny|"
        r"have sex with|sexual (act|contact|abuse|assault))\b", re.I)),

    # Drugs, alcohol, tobacco.
    ("substances", re.compile(
        r"\b(cocaine|heroin|meth|marijuana|weed|cannabis|vape|vaping|"
        r"get (high|drunk|stoned)|drugs? to|beer|vodka|whiskey|alcohol|cigarette)\b", re.I)),

    # Slurs / hate. Kept as category patterns rather than an explicit slur list.
    ("hate", re.compile(
        r"\b(nazi|hitler|kkk|white power|"
        r"(hate|kill|deport) (all )?(black|white|asian|mexican|jewish|muslim|gay|trans)\w*|"
        r"retard(ed)?|faggot|n[ -]?word)\b", re.I)),

    # Profanity.
    ("profanity", re.compile(
        r"\b(fuck\w*|shit\w*|bitch|asshole|bastard|cunt|dick(head)?|piss off)\b", re.I)),

    # Personal information — the protocol reminds participants not to share it.
    ("personal_info", re.compile(
        r"\b(my (home )?address|where i live|my phone number|my last name|"
        r"my school is|my password|social security|credit card)\b", re.I)),

    # Medical / mental-health advice — out of scope, refer to the adult.
    ("medical", re.compile(
        r"\b(should i take|is it safe to (take|swallow|drink)|"
        r"i (feel|am) (depressed|anxious|suicidal)|"
        r"diagnos\w*|prescri\w*|my medication|am i sick)\b", re.I)),

    # Prompt injection / role-play jailbreaks.
    ("jailbreak", re.compile(
        r"(ignore (all |your |previous |the )*(instruction|rule|prompt)|"
        r"disregard (your|the|all) (instruction|rule)|"
        r"you are (now|no longer) (a|an|my)|pretend (you are|to be) (a|an|my)|"
        r"system prompt|developer mode|jailbreak|do anything now|"
        r"forget (your|the|all) (rule|instruction))", re.I)),

    # Scope: the protocol turns off coding, image generation and browsing.
    ("off_scope", re.compile(
        r"\b(write (me )?(some )?code|write a (python|javascript|java) |"
        r"generate an image|draw me a picture of|browse the (web|internet)|"
        r"search (the web|online) for|what('s| is) your (system )?prompt)\b", re.I)),
]

# Applied to MODEL OUTPUT only — things the model should never say to a child even
# if the question was innocent.
ANSWER_RULES: list[tuple[str, re.Pattern]] = [
    ("answer_profanity", RULES[6][1]),
    ("answer_sexual",    RULES[3][1]),
    ("answer_self_harm", RULES[0][1]),
    ("answer_violence",  RULES[1][1]),
    # No trailing \b here: "call 555" ends on a digit followed by another digit,
    # where \b fails and the whole alternation silently stops matching.
    ("answer_contact", re.compile(
        r"(?:\bcall \d|https?://|\bwww\.|@[\w.]+\.(?:com|org|edu)\b|"
        r"\bemail (?:me|us)\b|\bphone number\b)", re.I)),
]


def _scan(text: str, rules) -> Verdict:
    for category, pattern in rules:
        m = pattern.search(text)
        if m:
            return Verdict(False, category, m.group(0))
    return Verdict(True)


def check_question(text: str) -> Verdict:
    """Gate a child's question before it reaches the model.

    Every rule applies regardless of animal context. There is deliberately no
    "it mentions a shark, so let it through" escape hatch: such an escape is
    trivially abused ("do sharks watch porn"). Instead each pattern is written
    narrowly enough that marine-biology talk never trips it — "does the octopus
    kill the crab" does not match the violence rule because that rule requires a
    human object. The MUST_PASS battery in tests/test_safety.py is what keeps
    that property honest; run it after touching any pattern.
    """
    text = (text or "").strip()
    if not text:
        return Verdict(False, "empty", "")
    return _scan(text, RULES)


def check_answer(text: str) -> Verdict:
    """Gate model output before the experimenter sees it."""
    return _scan(text or "", ANSWER_RULES)


# ── Institution scrubbing ────────────────────────────────────────────────────
# Belt-and-braces for the system-prompt rule: answers describe the animal, not any
# particular aquarium. Drops whole sentences that name an institution or speak as one.
_INSTITUTION = re.compile(
    r"(birch|scripps|aquarium\.ucsd|ucsd|our aquarist|our aquarium|our tank|"
    r"here at (the|our)|come (and )?visit|when you visit|on display here|"
    r"in our (tank|exhibit|care)|we (have|keep|care for))", re.I)


def strip_institution(text: str) -> tuple[str, bool]:
    """Remove sentences that reference a specific institution. Returns (text, changed)."""
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    kept = [p for p in parts if p and not _INSTITUTION.search(p)]
    out = " ".join(kept).strip()
    if not out:                      # never blank the whole answer
        return text.strip(), False
    return out, out != text.strip()
