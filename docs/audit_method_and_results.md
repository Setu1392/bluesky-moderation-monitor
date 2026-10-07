# Hand-audit: how accurate is the model on Bluesky?

## The question

The model was trained on Wikipedia talk-page comments and tested on held-out Wikipedia comments, where its strict mode reached 93.7% precision. Bluesky replies are a different kind of text, so that number says little about live performance. This audit measures **precision on Bluesky**: of the replies the model flags, how many are actually bad?

It does **not** measure recall (what the model misses), which would need a much larger look at unflagged replies.

## Method

**Population.** English replies flagged in Sensitive mode while audit mode was on. Strict mode flags a subset of these (its thresholds are at least as strict for every label), so one sample answers both: the Strict figures use the replies in the sample that Strict mode also flagged.

**Sampling.** Reservoir sampling with a capacity of 100 (`python tox_phase3_stream.py --audit 100`). Every flagged reply seen during the run has the same chance of being in the final sample, and at most 100 texts are ever stored. The text is held in a separate local file (`tox_audit.db`), cleaned the same way the model saw it (mentions and URLs removed), with no account names. Posts deleted on Bluesky are removed from the sample.

**Review.** One reviewer (me), using `python tox_audit.py`:

- *Blind*: the model's scores and categories are never shown.
- *Shuffled*: the order is random, so stopping early still leaves a random subset.
- *No context*: each reply is shown without the post it answered.

Each reply gets one answer:

| Key | Answer | Definition |
|---|---|---|
| `a` | Abusive | Insults, harasses, threatens, demeans or attacks a person or group |
| `p` | Profane only | Crude or swearing language that is not aimed at attacking anyone |
| `n` | Not toxic | Neither of the above, including harsh disagreement about ideas |
| `s` | Skip | Can't tell, too ambiguous, or declined |

All 100 replies received an answer (0 skipped). When the review finishes, the tool offers to delete the stored comment text (or run `python tox_audit.py --purge` at any time); labels and scores are kept.

**Statistics.** 95% Wilson score intervals, which are reliable at small sample sizes and for rates near 0% or 100%, unlike the textbook ±1.96σ formula.

## Results

| | Reviewed | Abusive | Profane only | Not toxic |
|---|---|---|---|---|
| **Strict mode** | 56 | 15 (27%, CI 17–40%) | 23 (41%) | 18 (32%, CI 21–45%) |
| **Sensitive mode** | 100 | 17 (17%, CI 11–26%) | 33 (33%) | 50 (50%, CI 40–60%) |
| *Sensitive only (flagged in Sensitive, not Strict)* | 44 | 2 (5%, CI 1–15%) | 10 (23%) | 32 (73%, CI 58–84%) |

Abusive *or* profane: Strict 38 of 56 = **68%** (CI 55–79%); Sensitive 50 of 100 = 50% (CI 40–60%); Sensitive-only band 12 of 44 = 27% (CI 16–42%).

*(The Strict and band counts are recovered from the printed percentages, which have a unique integer solution at these sample sizes; the totals match `audit_summary.json`.)*

**Reading the results.**

1. **Precision on Bluesky is far below the Wikipedia figure.** 27% of strict flags were abusive and 68% were abusive or profane, against 93.7% on Wikipedia. The 68% is the fairer comparison, because Wikipedia's "toxic" label also covers crude language, but the gap is still large.
2. **The score still ranks sensibly.** Raising the threshold from Sensitive to Strict took the abusive rate from 17% to 27% and cut false alarms from 50% to 32%. The extra replies Sensitive mode adds are 73% false alarms, which is why the dashboard defaults to Strict.
3. **Implied share of the whole stream.** With Strict mode flagging 7.3% of English replies in the dashboard window I checked (7–8% across sessions), about **2.0%** of replies (1.2–2.9%) were flagged *and* judged abusive, and about **5.0%** (4.0–5.7%) were flagged and judged abusive or profane. These are floors, because recall is unmeasured, and the flag rate and the audit sample come from different time windows.

## Limitations

- **One reviewer.** My threshold for "abusive" is mine; a second reviewer would let me report agreement (Cohen's kappa).
- **No parent post.** Without the post being answered, some replies are ambiguous. Sarcasm and banter are easy to misjudge in either direction.
- **Small samples.** Strict mode has 56 reviews, so its interval is about ±12 points.
- **Precision only, one session.** Time of day, news events and conversation topics all shift what people post.
- **Different label definitions.** My three-way scheme is not the Jigsaw annotators' definition of "toxic", so the comparison with the Wikipedia figure is approximate.

## Reproduce it

```bash
python tox_phase3_stream.py --audit 100     # let it run 30+ minutes, then Ctrl+C
python tox_audit.py                          # review; writes audit_summary.json
python tox_phase4_dashboard.py               # the Limitations panel now shows the results
```

Never commit `tox_audit.db`: it holds comment text until you purge it. `audit_summary.json` contains counts and rates only and is safe to share.
