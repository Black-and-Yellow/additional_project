# Reviewed implementation, 15 November 2026

## Current release

The source now uses retrieval by default for model runs. The controlled comparison did not show an adaptive accuracy advantage, so adaptive gating remains an explicit experimental option. The always-retrieve path now goes directly to retrieval and contextual generation, removing an unused context-free answer.

Evidence verification checks every sentence. The review adds a conservative numeric guard before NLI: complete time pairs stay together, leading zeros normalize, and unsupported numbers reject a candidate. Common titles such as “Dr.” no longer split a sentence. These checks can reject legitimate paraphrases and do not prove that every relation in a claim is correct.

The fitted profile now validates schema, dimensions, finite coefficients, positive scales, and valid component indexes before use. Generation-token accounting includes the first generated EOS and excludes padding after it.

## Verification

- 40 regression tests passed during the review, covering control flow, persistence, input validation, evidence guards, split leakage, scoring, and fitted-profile validation.
- A real-model smoke check returned the supported Canberra answer in one generation call, rejected a mismatched opening time, and loaded the calibrated tensor detector. See `reports/release-verification.json`.
- Browser checks exercised the current lookup interface and mobile layout. No page errors or horizontal overflow appeared in that check.
- The PowerPoint includes an application screenshot, native editable tables and a latency chart, references, and speaker notes. Its results correspond to the recorded experiments, with timing limitations disclosed.

## Detector results

The frozen backbone is SmolLM2-135M-Instruct. A logistic detector learns from External Context Score and Parametric Knowledge Score. Candidate component selection uses the training partition, and threshold choice uses the validation partition. The supplied fitted profile selects layer 26, head 1.

The data comprises 64 fictional handbook claims, split into 32 training, 16 validation, and 16 test claims. Source groups are disjoint across partitions. The test results are AUROC 0.734, F1 0.50, precision 0.75, recall 0.375, and accuracy 0.625. The detector misses five of eight contradictory test claims. These modest results concern controlled context consistency rather than general hallucination detection.

## Historical comparison, 13 November

| Policy | Opening-time check | Mean GPU latency | Total generated tokens |
| --- | --- | --- | --- |
| No retrieval | 0/8 | 1.11 s | 219 |
| Always retrieve | 8/8 | 1.46 s | 332 |
| Adaptive, correction off | 8/8 | 3.14 s | 1116 |
| Adaptive, guarded | 8/8 | 3.12 s | 1116 |

All historical configurations included initial context-free generation. The reviewed always-retrieve path removes that generation, so the table is not a current-release timing benchmark. Historical token counts excluded EOS, whereas the reviewed counter includes the first EOS. The original reports remain unchanged for traceability.

No intervention triggered on those eight test questions. The comparison supports retrieval on that narrow task, shows adaptive overhead, and gives no evidence of correction benefit. The opening-time metric combines extraction and an NLI proxy; it is not a comprehensive correctness assessment.

## Boundaries

The component selection and intervention policy are project adaptations inspired by ReDeEP. The current implementation does not reproduce the paper's full benchmark procedure. A larger independent dataset and model comparison are needed before claiming generalization or improved hallucination mitigation.
