# Responsible AI Checklist — Clinical Context

This checklist maps the Day 4 clinical-safety requirements to the implemented MedFlow controls. The team should review it together before the live demo.

- [x] **No answer implies it replaces clinical judgment.** Day 4 diagnostics expose a visible disclaimer: MedFlow is guideline-grounded clinical decision support and does not replace individualized clinical judgment or professional medical evaluation.
- [x] **Uncertainty language matches evidence strength.** `safety_guardrails.py` maps strong / partial / weak / insufficient evidence to different wording and deterministic refusal behavior.
- [x] **Refusals are never softened for the demo.** Below-threshold, unsupported numerical claims, low faithfulness, or failed citation accuracy trigger the structured `insufficient` refusal path.
- [x] **Disclaimer is visible, not buried.** `day4_pipeline.ask_safe_clinical_question()` returns the safety diagnostics envelope by default, including the disclaimer and guard reason(s).
- [x] **Unsupported-claim check is independent of the generation prompt.** `claim_validator.py` runs after generation.
- [x] **Safety choices are auditable.** Threshold sweep, per-query results, guard reasons, citation details, and faithfulness details are serializable in `results/day4/`.

## Team sign-off before Day 5

- [ ] Run the full evaluation against the audited exact frozen Day 2 index.
- [ ] Review all false refusals and any unsafe accepts.
- [ ] Confirm mean faithfulness is at least 0.90 before quoting it.
- [ ] Confirm every presented metric comes from the saved evaluation artifacts, not illustrative notebook examples.
- [ ] Rehearse one supported answer and one below-threshold refusal in the live demo.
