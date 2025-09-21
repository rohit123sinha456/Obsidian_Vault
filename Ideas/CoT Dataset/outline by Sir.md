=======================
POSSIBLE TITLES
=======================
* Mind's Eye: A Benchmark of Visual Abstraction, Transformation and Composition for Multimodal LLMs


=======================
OTHER CONSIDERATIONS
=======================
* Do we want to provide any code during review process?
* Do we want to provide any dataset samples as a ZIP file (or on an anonymized website)?
* Do we want to do any finetuning to say how models behave if given this option?

=======================
POSSIBLE OUTLINE
=======================
1. Introduction
    * Para 1: Motivation: gap between object recognition and human-like visual reasoning in multimodal LLMs
    * Para 2: What we have built (dataset/tasks) and why it fills a gap (link to cognitive literature)
    * Teaser figure on top of pg 2: 
        A single mosaic image showing one canonical example from all 8 tasks (like we have currently, we may need to improve this). We can show: (i) ground truth answer, (ii) the most common model failure (small red overlay), (iii) human answer (if we have). Would be nice to communicate task diversity and failure modes
    * Para 3: Short summary of key findings (headline numbers, e.g., models at ~X vs human P)
    * List of contributions
    
2. Related Work
    * Multimodal foundation models
    * Current multimodal benchmarks (consider adding CLEVR, RAVEN, ARC, RPM apart from what we have currently written up), what they measure, we need to show how our tasks are complementary and more fine-grained for spatial/transformational abilities -- this can subsume the broader ones and cognitive ones as two paras of this side-heading
    * Multimodal LLMs evaluation literature (do we have anything here apart from benchmarks?)
    * Inspiration from cognitive science
    
3. The Mind's Eye Benchmark
    * Overlay of the benchmark -- Carroll's Three-Stratum Theory, Gf (fluid intelligence), tasks under Gf and how our 3-level ladder view (Abstraction → VRA, HPE; Relations → DSC, VCS, SS; Transformations → MT, PF, MC) maps to tasks under Gf
        * Diagram (see picture shared on chat for how we could do this)
    * Task definitions and generative procedure for each task (design choices, parameters, difficulty levels, train/val/test split, statistics)
    * Psychometric design: how we ensured variety, controlled confounders (e.g. ensured that a model or user cannot solve it using some shortcut), calibration of difficulty (we can add item response curves here)
        - How to plot item response curves:
            * Plot probability of correct response as a function of ability 
            * For humans, define ability from test scores (can use IRT models)
            * For models, we can definie ability in terms of either model size (#params), overall accuracy across tasks, or logit confidence.
        - Please see later part of this outline to see other kinds of item response curves we can plot
    * Human baseline protocol (how many subjects, instructions, time limits) -- depending on what we can pull off here
        - We may need at least 30–50 human responses per item or per split to get stable baselines and item difficulties

4. Experimental Setup and Results
    4.1 Models and Experimental Setup
        * Models evaluated (open-source + API-based), input pipeline, prompting details (zero-shot, few-shot, CoT), image pre-processing?
        * Evaluation metrics (accuracy, chance baseline, human ceiling, per-skill score?)
    4.2 Results
        * Table like we already have (per-task performance). Add human baselines.
            - Add mean and standard deviation for each entry (preferably for 5 trials, or at least 3 trials)
            - Group columns to represent our ladder of tasks (we can add aggregates of task-level performance if we want too)
            - Use bold for top performance
        * Possible figure: Model performance heatmap -- models (rows) x tasks (columns) with color-coded accuracy (with human and random baseline rows). We could organize the rows in increasing order of model capability, and columns in increasing order of the ladder levels
        * Possible figure: Skill-level bar chart: group tasks by cognitive level (3 diff ladder levels we have) and show average performance per level per model (with human ceiling)
        * Short interpretation paragraph per task group (taxonomy levels)
        * **Qualitative CoT analysis: for each task, show 2–3 cases (one success, one failure) with model answers, reasoning traces and a short explanation/hypothesis (We can add 2-3 tasks here for some models and rest in the appendix)

5. More Results and Analysis
    * Results with different prompting strategies (incl ones that encourage spatial reasoning)
        - Table or line plots showing effect of prompting style (zero-shot, few-shot, CoT) on per-task performance.
        - Per-task effect: Report task-wise deltas (Will highlight where CoT helps)
    * (TO DO) Results with diff vision encoder variants?
        - run a small test with two different image encoders for same LM head and compare (If failures persist across encoders, problem is reasoning)
    * Analysis of model scaling (Qwen 3B -> 7B -> 32B) -- please see end of outline on how we could do this
        * Possible figure: Scatter plot of model size/compute vs task performance: x = model parameter count (or FLOPs / multimodal pretraining scale), y = overall benchmark score or tiered score. We can show scaling trends or also lack thereof
    * (TO DO) Results with same architecture but trained on larger vision-text datasets; can we do this?
    * **Per-task error analysis: A bar chart or table where for each task, for each model (or we can show this for 1-2 models or Qwen family alone), we show total errors, errors when the closest answer is chosen, and errors when the other 2 answers are chosen. This can help show which distractors models pick most often — reveals systemic model biases.**
    * Probing internal representations: Train linear probes on visual encoder activations to predict intermediate symbolic features (e.g., number of folds in mental folding task). If probes succeed, the encoder has the info but the LM doesn’t use it.

6. Discussion
    * What the failures imply about current model architectures (e.g., models may rely on pattern matching, not transformation)
    * Suggestions for future models (structured symbolic modules, visual reasoning modules, better pretraining?)
    * Limitations of our benchmark

7. Conclusions and Future Work

===========
APPENDIX
===========
A1. Full dataset specification -- details, more sample images for each task
A2. More qualitative results (expanded from Sec 4.2, can be comprehensive here)
A3. More studies
    A.3.1. Item perturbation / OOD: random rotation, noise, distractor injection — how fragile are the models?
A4. Model visualizations
    Show predicted answer + saliency map (if we can get attention from model or GradCAM from vision encoder)
A5. Human experiment details
A6. Full prompt templates
code link?
    
==================================================
Plotting item response curves (Got from ChatGPT)
==================================================
Option a -- Across models
	* Treat each model (Qwen-7B, GPT-4o, etc) as a participant with some ability θ.
	Define θ as model’s average benchmark accuracy (or logit-based confidence if available).
	For each item (or task family), plot:
	x-axis: θ (ability).
	y-axis: probability that the model answers that item correctly.
	Fit a logistic curve (1-parameter or 2-parameter logistic IRT model):
P(\mathrm{correct}\mid\theta)=\frac{1}{1+e^{-a(\theta-b)}}
	b = difficulty parameter (θ where 50% chance of success).
	a = discrimination parameter (slope, how sharply item separates low/high θ).
This lets you show: “Paper Folding items have higher difficulty (b) than Symmetry items; only high-ability models solve them.”
________________________________________
Option B — Within models across difficulty levels
	If you’ve generated task items at different difficulty settings (e.g., # of folds in PF, # of rotation steps in MT), you can treat difficulty index as θ.
	Then, plot probability correct (y-axis) vs difficulty parameter (x-axis).
	This gives empirical IRCs for each task — showing whether accuracy drops smoothly with difficulty or collapses abruptly.
________________________________________
Option C — Human vs model comparison
	Collect human responses → estimate human θ distribution via standard IRT.
	Overlay model curves (using model accuracy as θ proxy).
	This shows how “human-like” the difficulty progression is (e.g., humans show graceful slope; models show step-function failures).

==================================================
Plots for model scaling (with help from ChatGPT)
==================================================
Consider Qwen-VL 3B, 7B, 14B
Plot per-task accuracy vs model size (log-scale).
X-axis = parameter count (log scale).
Y-axis = accuracy (for each task).
Use separate curves for task categories (Abstraction / Relations / Transformations).
Add human ceiling as dashed line
Fit a simple log-linear trend:
\text{Accuracy(task)} = A + B \cdot \log(\text{#params})
See whether performance improves smoothly with size or shows a plateau.
Compare across task levels.
    * E.g. Scaling may help perceptual/relational tasks but not transformational imagery (mental rotation, paper folding). This  asymmetry may be nice to report.

======================================
SOME GENERAL CONSIDERATIONS
========================================
* Add human performance as a horizontal dashed line in bar charts wherever possible
* Report error bars/confidence intervals for plots/tables wherever possible
* Use consistent color palettes; order tasks by cognitive level in all plots so patterns are readable.
* Include small insets with sample images near plots that refer to particular data points (e.g., place a sample image next to a confusion cell that corresponds to the most common error)





Teaser image
Vision tower abalation
write up the plots interpretation
increate the main table with smaller models
in appendix include the tables for all type pf promtps ( each prompt type one table)
add random base line in task specific plots as dotted lines