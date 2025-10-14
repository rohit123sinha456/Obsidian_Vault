

---
You: Following the idea of subsection 3.1 an 3.2 give some novel methods that probes the models performanc.

---
Moonlight: Nice — below I propose a suite of novel, practical probing methods inspired by subsection 3.1 (Option Order Sensitivity Test) and subsection 3.2 (Slot Guessing for Perturbed Caption). For each method I give the intuition, a clear protocol, the metric(s) to compute (extensions of $CR$, $PCR$, $\Delta$, $\Phi$), likely confounders/failure modes, and simple mitigations. I finish with suggestions for experiments to validate and calibrate thresholds and some follow-up research questions.

Notation (used below):
$$
\begin{aligned}
CR &= \frac{N}{|D|} \\
PCR &= \frac{N'}{|D|} \\
\Delta &= PCR - CR \\
\Phi &= \frac{X}{|D|}
\end{aligned}
$$
Here $N$ and $N'$ are counts of correct answers before and after perturbation, $|D|$ is dataset size, and $X$ counts instances that flip from correct to incorrect after perturbation.

Novel probing methods

1) Distractor Semantic Perturbation Test (DSPT)
- Intuition: Option-order sensitivity tests position-based shortcuts; DSPT probes whether a model relies on superficial lexical cues in distractors rather than semantic reasoning by systematically replacing distractors with semantically closer or more distant alternatives.
- Protocol:
  1. For each multiple-choice instance $(Q,I,A=\{a_1,...,a_m\})$, compute semantic similarity between each distractor $a_j$ and the correct answer $a_c$ using an embedding model (e.g., SBERT) to get similarity score $s_j$.
  2. Create two perturbed sets: (i) Close-distractors: replace each distractor by another option with high $s_j$ (semantically close), (ii) Far-distractors: replace distractors with low-$s_j$ options (semantically distant) sampled from a pool of unrelated answers.
  3. Evaluate the model on both perturbed sets and compute $CR_{close}$ and $CR_{far}$.
- Metric: report $\Delta_{close} = CR_{close}-CR$ and $\Delta_{far} = CR_{far}-CR$ and asymmetry $A = \Delta_{close}-\Delta_{far}$.
- What it reveals: a large positive $A$ indicates model performance collapses when distractors are too similar (suggesting weak fine-grained reasoning), whereas a large negative $A$ may indicate exploitation of distractor implausibility shortcuts or memorized patterns.
- Confounders & mitigations: selection of replacement distractors can accidentally introduce new cues; control by sampling multiple replacement pools and reporting mean ± CI.

2) Cross-Instance Option Substitution Test (COST)
- Intuition: If a model memorizes question–answer pairs (option-text links), substituting the entire option set from other instances should break such memorization more than robust reasoning.
- Protocol:
  1. For a batch of $k$ instances, scramble answer-option sets across instances while keeping each instance’s question and image intact.
  2. Evaluate original $CR$ and substituted $CR_{sub}$ and compute $\Delta_{sub} = CR_{sub}-CR$.
- Metric: $\Delta_{sub}$ and per-instance flip rate $\Phi_{sub}$.
- What it reveals: strong drop indicates reliance on instance-level memorized option mappings.
- Confounders: if new substitute options are implausible, drop could come from unnatural distractors; address by constraining substitute options to same semantic category (e.g., same question type or topic).

3) Option-Context Decoupling / Anchored-Answer Test (OCAT)
- Intuition: Checks if the model depends on relative phrasing/position of the correct option within the overall prompt context (e.g., punctuation, numbering).
- Protocol:
  1. For each instance create multiple anchor formats: enumerated answers (A., B., ...), bulleted answers, inline answers separated by semicolons, and an answer-only line.
  2. Compute $CR$ for each anchor format and the variance $\sigma^2_{format}$ across formats.
- Metric: format sensitivity measured by $\sigma_{format}$; normalize by chance level.
- What it reveals: high sensitivity suggests brittle prompt-format dependencies or memorized prompt templates.
- Mitigations: average across many formats, use permutation test to estimate significance.

4) Contextual Cloze Robustness Test (CCRT) — extension of Slot Guessing
- Intuition: Instead of masking a single keyword, remove several context words from the question (or caption) to test whether the model relies on exact phrasing of the prompt rather than image grounding or deeper semantics.
- Protocol:
  1. For each sample mask $k$ tokens from the question/caption chosen by importance (e.g., highest TF-IDF or gradient saliency). Vary $k$ across $\{1,2,3\}$.
  2. Create paraphrased versions for each masked prompt (via back-translation or LLM paraphrase).
  3. Evaluate with image present and absent.
- Metric: track $CR_{mask,k}$ and $CR_{paraphrase,k}$; define per-instance cloze fragility $F_i = \frac{1}{K}\sum_k \mathbf{1}[\text{correct on original}\ \wedge\ \text{incorrect on paraphrase}]$.
- What it reveals: if the model answers correctly only on original wording even when visual evidence exists, this is evidence of memorized surface forms.
- Confounders: many visual tasks remain solvable with part of the question; use image-absent control to separate unimodal leakage.

5) Image-Text Mismatch / Swap Test (ITST)
- Intuition: If a model memorized $(Q,I)\to A$ triplets, swapping images between different questions that are semantically incompatible should reduce performance; if model uses only text, swapping images will not affect accuracy.
- Protocol:
  1. For a batch, swap images across instances in a way that preserves general domain but breaks semantic alignment (e.g., swap images within same scene class but different objects).
  2. Compute $CR_{swap}$ and $\Delta_{swap}=CR_{swap}-CR$.
  3. Also run the complementary test: replace image with a semantically similar but non-identical image (to measure generalization vs memorization).
- Metric: $\Delta_{swap}$ and per-instance flip fraction $\Phi_{swap}$.
- What it reveals: small $\Delta_{swap}$ but high unimodal LLM accuracy indicates unimodal contamination; large $\Delta_{swap}$ indicates models rely on image-language alignment (good) or memorized $(Q,I)$ indexes (bad) depending on pattern.
- Confounders: some questions are purely textual; filter for image-reliant instances via human annotation or automated heuristics.

6) Progressive Lexical Corruption Test (PLCT) for captions
- Intuition: Gradually corrupting caption surface forms (character swaps, synonym replacement, word-dropout) reveals whether the model uses exact lexical matches or is robust to paraphrase.
- Protocol:
  1. For each caption generate levels $L=0..L_{max}$ of corruption where at level $L$ a fraction $p_L$ of tokens are replaced by random characters or synonyms.
  2. Evaluate and plot $CR(p)$ as a function of corruption $p$.
- Metric: robustness slope $s = \frac{d CR}{dp}$ and area under the robustness curve $AURC$.
- What it reveals: a steep negative slope indicates lexical memorization; gentle slope indicates semantic grounding.
- Confounders: very aggressive corruption can create unnatural prompts; use paraphrase-based corruption instead of random noise to preserve plausibility.

7) Cross-modal Retrieval Consistency Test (CRCT)
- Intuition: If a model has memorized image-caption pairs, it should be able to retrieve the original caption or ID given either modality; test retrieval consistency as a proxy for memorization.
- Protocol:
  1. Ask the model to identify which of $k$ candidate captions best describes the image (closed-set retrieval) where $k$ includes the true caption plus near-neighbor captions from a large pool.
  2. Run the reverse: provide the caption and ask which of $k$ images matches.
- Metric: retrieval accuracy $R_{img\to cap}$ and $R_{cap\to img}$ and asymmetry.
- What it reveals: extremely high accuracy on retrieval with identical captions may indicate memorization; failing on paraphrased captions indicates reliance on surface text.
- Confounders: semantic overlap between candidate captions; mitigate with careful candidate selection (hard negatives).

Composite scoring and statistical testing
- Per-instance contamination score: combine binary signals from multiple tests into a continuous score, e.g.,
$$
S_i = w_1 \cdot \mathbf{1}[\text{flip in DSPT}_i] + w_2 \cdot \mathbf{1}[\text{flip in ITST}_i] + \dots
$$
where $w_j$ are weights (or use logistic regression to combine signals).
- Population-level significance: for any $\Delta$ estimate compute permutation-based null distribution by randomizing perturbations (e.g., shuffle mapping between perturbed and unperturbed versions) to obtain p-values and confidence intervals.
- ROC-style evaluation: when you synthetically inject contamination (ground truth), treat detection as binary classification and compute AUC to calibrate thresholds.

Practical validation experiments
- Synthetic contamination injection: inject known fractions $\rho\in\{0,0.1,0.5,1.0\}$ of benchmark samples into training of a toy MLLM to measure sensitivity of each probe and calibrate detection thresholds.
- Cross-model comparison: run probes across open-source and closed models to measure relative susceptibility; correlate probe scores with known dataset overlaps (when available).
- Ablations: test how many paraphrases, shuffles, and corruption levels are needed to stabilize metrics; report standard errors.

Failure modes and best-practice mitigations (summary)
- False positives from poor paraphrases or unnatural distractors: use semantic-similarity filters (SBERT) and multiple paraphrases per instance.
- False negatives from very robust models: increase perturbation difficulty (mask multiple slots, use hard negatives) and combine probes.
- Nongeneralizable thresholds: always calibrate using synthetic injection and report effect sizes with CIs.

Follow-up research directions
- Learn-to-detect: train a small detector that ingests multi-test signals (DSPT, ITST, PLCT, etc.) and predicts contamination probability, then test generalization to unseen datasets and models.
- Causal tracing of contamination origin: combine unimodal LLM probing (as in §6.1) and cross-modal tests to disentangle unimodal vs cross-modal leakage systematically.
- Extend to other modalities: adapt ITST and PLCT ideas to audio/video (swap audio tracks, progressively corrupt audio transcripts, etc.).
- Theoretical analysis: characterize conditions under which a model can pass all perturbation tests yet still be contaminated (tight worst-case guarantees).

If you want, I can:
- Provide ready-to-run pseudocode for 2–3 of these probes (DSPT, ITST, PLCT).
- Propose an experimental plan (datasets, model sizes, injection schedules) to benchmark and compare probes.
- Implement a scoring rubric combining the probes into an aggregated contamination index.

Which of these would you like to develop next?

---
You: give some more novel methods that probes the models performance.

---
Moonlight: Good — below are ten additional, novel probing methods (beyond the earlier list) to stress-test multimodal models. For each I give intuition, a clear protocol, the metric(s) to compute (extensions of $CR$, $PCR$, $\Delta$, $\Phi$), main failure modes, and simple mitigations. I assume the usual atomic metrics:
$$
\begin{aligned}
CR &= \frac{N}{|D|} \\
PCR &= \frac{N'}{|D|} \\
\Delta &= PCR - CR \\
\Phi &= \frac{X}{|D|}
\end{aligned}
$$
where $N$ and $N'$ are counts of correct answers before and after perturbation, $|D|$ is dataset size, and $X$ counts instances that flip from correct to incorrect after perturbation.

1) Counterfactual Image Editing Test (CIET)  
- Intuition: If a model truly grounds answers in image content, making a minimal semantic edit to the image that flips the correct label should flip the model’s answer; memorized text pairs won’t adapt.  
- Protocol: apply a small, targeted edit (e.g., replace a red ball with a blue cube) using image-editing or synthetic overlays for each image and evaluate $CR_{edit}$. Compute $\Delta_{edit}=CR_{edit}-CR$.  
- Metrics: report $\Delta_{edit}$ and flip rate $\Phi_{edit}$.  
- Failure modes: edits might be imperceptible or change unrelated features; mitigate by using human-verified edits or automated saliency-guided edits to ensure the edited region is causally relevant.  
- Experiments: vary edit strength and measure the monotonic relation between edit magnitude and $\Delta_{edit}$.

2) Temporal Consistency / Time-Shifted Prompt Test (TCPT)  
- Intuition: Many datasets contain temporal cues; if a model memorized a dataset snapshot, asking the same multimodal question but with a time-relative phrasing (e.g., “previously”, “next year”) probes whether the model uses memorized static mappings.  
- Protocol: create time-shifted variants of questions (e.g., change “is the traffic light red?” to “was the traffic light red earlier?”) and measure $CR_{time}$ and $\Delta_{time}=CR_{time}-CR$.  
- Metrics: $CR_{time}$ and per-instance temporal flip $\Phi_{time}$.  
- Failure modes: some questions are time-independent; filter for temporally sensitive instances (automated heuristics or human labeling).

3) Semantic Role Swap Test (SRST)  
- Intuition: Swap semantic roles in the question/caption (agent vs patient, subject vs object). Robust reasoning should update answers accordingly; memorized surface-text matches may not.  
- Protocol: generate role-swapped prompts (e.g., “What is the man holding?” → “What is holding the man?”) and measure $CR_{role}$ and $\Delta_{role}=CR_{role}-CR$.  
- Metrics: $\Delta_{role}$ and instance-level role-sensitivity score.  
- Failure modes: grammatical nonsense after swap; mitigate with LLM-based rewriting constrained to preserve grammaticality.

4) Multi-Granularity Masking Test (MGMT)  
- Intuition: Mask text at different granularities (character, subword, word, phrase) in the caption/question; if the model depends on exact surface tokens, performance will drop earlier at finer granularity.  
- Protocol: for each sample produce corruption levels $p\in\{0.1,0.2,\dots\}$ at each granularity and measure $CR(p)$; compute robustness curve and slope $s=\frac{d\,CR}{dp}$.  
- Metrics: robustness slope $s$ and area-under-curve $AUC_{robust}$.  
- Failure modes: character-level noise may create unreadable prompts; prefer linguistically plausible perturbations (synonym replacement, morphological variants).

5) Anchorless Answering / Free-form Response Test (AFRT)  
- Intuition: Convert multiple-choice problems to open-ended prompts requiring the model to produce the label text instead of selecting an option; if the model learned to pick positions or templates, free-form performance should drop.  
- Protocol: transform each instance so answer choices are removed and the model must produce the answer token/string; compute $CR_{open}$ and compare with original $CR$ via $\Delta_{open}=CR_{open}-CR$.  
- Metrics: $\Delta_{open}$ and normalized string-match metrics (exact-match or normalized edit distance).  
- Failure modes: evaluation of free-form text is hard; use canonicalization (lowercasing, lemmatization) and an automated fuzzy-match pipeline.

6) Multiview Consistency / Multi-Angle Image Test (MCAT)  
- Intuition: Provide multiple views of the same scene (cropped, rotated, different zooms) and test whether answers are consistent; memorization of a specific image file often fails under viewpoint changes.  
- Protocol: for each instance supply $k$ views and measure per-view correctness $CR_j$ and consistency score $C_{cons} = \frac{1}{|D|}\sum_i \mathbf{1}[\text{all views agree}]$.  
- Metrics: average per-view $CR$, $C_{cons}$, and disagreement rate.  
- Failure modes: models may be sensitive to view-dependent occlusion even if grounded; choose multiviews that preserve critical content using detection or saliency.

7) Latent Activation Fingerprinting & Perturbation (LAFP)  
- Intuition: Compare internal representation fingerprints of model outputs on benchmark instances vs paraphrased/perturbed variants; a memorized instance will produce nearly identical activations for original prompts, while paraphrase-generalized models will change smoothly.  
- Protocol: collect latent activations (e.g., final-layer embeddings) for original and perturbed prompts; compute cosine similarity distributions and define fingerprint stability $S = \mathbb{E}[\cos(h_{orig}, h_{pert})]$.  
- Metrics: $S$ and activation gap $\Gamma = \mathbb{E}[\|h_{orig}-h_{pert}\|]$.  
- Failure modes: need white-box access; if unavailable, use output logits or embedding probes from accessible API.  
- Mitigation: perform in models where internals are accessible or approximate via hidden-state probing on open checkpoints.

8) Hard Negative Distractor Injection (HNDI)  
- Intuition: Insert human- or model-crafted hard negatives as distractors (semantically plausible alternatives) to see if models rely on easy heuristics or memorized answer-text alignment.  
- Protocol: for each question replace distractors with top-$k$ semantically-similar candidates found via retrieval (SBERT nearest neighbors) and measure $CR_{hard}$ vs $CR$. Compute $\Delta_{hard}=CR_{hard}-CR$.  
- Metrics: $\Delta_{hard}$ and per-instance hardness-sensitivity.  
- Failure modes: if distractors accidentally match correct answer, filter using semantic similarity thresholds.

9) Membership Retrieval via Prompted Completion (MRPC)  
- Intuition: Probe whether a model can complete sentence fragments from benchmark captions/questions better than generalization baseline; unusually high completion probability suggests memorization/membership.  
- Protocol: for each caption create masked prefixes/suffixes and query the model to complete them under controlled decoding (low temperature). Compare perplexity or token-prob of ground-truth continuation to a background distribution from random web captions; compute z-score $Z = \frac{\log p_{gt} - \mu_{bg}}{\sigma_{bg}}$.  
- Metrics: per-instance membership score $Z$ and fraction exceeding threshold.  
- Failure modes: LLMs are fluent and may assign high probability to plausible continuations; mitigate by calibrating background model on in-domain but non-overlapping corpus.

10) Cross-Dataset Transfer Probe (CDTP)  
- Intuition: Evaluate the model on semantically equivalent instances taken from different datasets; if performance collapses on the other dataset variants, the model may have memorized dataset-specific surface cues rather than generalizable semantics.  
- Protocol: assemble semantically matched pairs $(x_D,x_{D'})$ across datasets (e.g., same scene description but different captions); compute $CR_D$ and $CR_{D'}$ and transfer gap $G=CR_{D'}-CR_D$.  
- Metrics: transfer gap $G$ and asymmetry indicating dataset-specific memorization.  
- Failure modes: true domain shift will produce gaps; control with careful matching and human verification.

Practical recommendations for using these probes together
- Ensemble scoring: combine signals into an aggregated contamination/fragility score $S_{agg}$ via logistic regression or simple weighted sum; calibrate weights using synthetic contamination injections.  
- Statistical testing: for each metric compute permutation-based $p$-values by randomizing perturbations to obtain null distributions.  
- Synthetic calibration: inject known fractions $\rho$ of benchmark examples into a small fine-tune to measure each probe’s sensitivity and false-positive rate as a function of $\rho$.

Research directions and experiments I recommend next
- Benchmark the probes above on a diversified model suite (small open checkpoints to closed APIs) and on multiple datasets (multi-choice, captions, VQA). Measure detection AUC using synthetic injection as ground truth.  
- Ablation studies to find minimal perturbation budgets that reliably detect contamination versus the model generalization threshold.  
- Extend LAFP to learn a small classifier on activation-space that flags likely-leaked instances and evaluate cross-model generalization.

If you want, I can (pick one):  
- produce pseudocode for any 2 probes (e.g., CIET and LAFP),  
- design a small experiment plan (datasets, model list, synthetic injection schedule, evaluation script outline), or  
- draft the equations and a combined scoring formula $S_{agg}$ with calibration details. Which do you want next?