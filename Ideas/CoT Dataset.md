Create a CoT Based dataset for infographics Dataset

For domain specific Visual CoT Dataset, proposed framework.
Finetune/in-context learning of an LLM on that domain knowledge ( Like safety manual or other safety related documents ). Perform Open vocabulary object detection on the domain dataset, and then use LLM based CoT reasoning dataset for model finetuning/ alignment.

In Scene generation graphs like GAQ, we can take those questions that has multiple hops ( Like more than n hops ). We can select n like that fast and slow thinking paper did. we take n=1 hop perform inference see if the a model can answer properly then we do n=2 hop and do the same and we n=k hop until k is the value where the model fails to give correct answer.

So in VoCot like dataset, For each reasoning step we can use an auxilarry LLM to check the textutal step and the crop of the image are grounded or not ( that is the textutal reason for that image crop is valid or not).  and for each step we can come up with like a grounding score. and at the end of reasoning we can check the cumulative grounding score to see if the reasoning is valid or not. 

Using a Critic CoT we can expand the VoCoT as such
{ "prompt": "...", "chain_of_thought": "...", "final_answer": "...", "grounded_steps": [true, true, false], // per-step grounding "coherence_score": 4, "faithfulness": "high" }


MMReason take a look
MME-CoT take a look
GFlowNet Take a look
Mulberry: Empowering MLLM with o1-like Reasoning and Reflection via Collective Monte Carlo Tree Search

Lets take the datasets from MME-CoT,MMMU, MMMUPro, MMStar, and M3CoT,VoCoT,NaturalBench,VisOnlyQA, [[GRIT]],[[BLINK]],[[CVBench]],[[SEED-Bench]],[[Spatial457 A Diagnostic Benchmark for 6D Spatial Reasoning of Large Multimodal Models|Spatial457]], VSI-100K. Find the common modality. During evaluation, we will only input open-ended textual questions while excluding the image to test whether the MLLMs can answer correctly without visual input. If the model can still answer correctly without any shortcuts or visual cues, we consider the instance potentially memorized or visually weakly relevant. Refere to the audio in whatsapp.

Most of these papers have MCQ based question or single answer based question. There is no Free form text answer.
 Grounding scoring: measure **IoU** or region descriptor accuracy against reference using grounding models
 Robustness Checks
 Introduce **visual distractor patches** or swap image parts
 Insert **adversarial region hints** in prompts; detect if models rely on these rather than true regions



Partial-CoT, given half the CoT steps can we regenerate the rest of the CoT correctly. If the models is goinf haywire it wont do that else it would.

Use Caption this read that paper dataset on VoCoT and check how good is its spatial reasoning task. What can we do for better spatial reasoing in CoT



CoT-Critic.


Combining The dataset of Caption this read that paper, those orientation image dataset, spatial reasoning Dataset, HRIBench ( Non-verbal cues)[https://arxiv.org/pdf/2506.20566]


### General Observation: 

These reasoning works are based on [Raven's Progressive Matrices](https://en.wikipedia.org/wiki/Raven%27s_Progressive_Matrices) which measures the [fluid intelligence](https://en.wikipedia.org/wiki/Fluid_and_crystallized_intelligence) However, test like Woodcock–Johnson Tests of Cognitive Abilities, Third Editiontalks about : 
1) Concept formation:- Individuals have to apply concepts by inferring the underlying "rules" for solving visual puzzles that are presented with increasing levels of difficulty. As the level of difficulty increases, individuals have to identify a key difference (or the "rule") for solving puzzles involving one-to-one comparisons. For more difficult items, individuals need to understand the concept of "and" (e.g., a solution must have some of this and some of that) and the concept of "or" (e.g., to be inside a box, the item must be either this or that). The most difficult items require fluid transformations and [cognitive shifting](https://en.wikipedia.org/wiki/Cognitive_shifting "Cognitive shifting") between the various types of concept puzzles that the examinee had worked with previously.
2) Analysis–Synthesis:- In the Analysis–Synthesis test, the individual has to learn and orally state the solutions to incomplete logic puzzles that mimic a miniature mathematics system. The test also contains some of the features involved in using symbolic formulations in other fields such as chemistry and logic. The individual has presented a set of logic rules, a "key" that is used to solve the puzzles. The individual has to determine the missing colors within each of the puzzles using the key. Complex items presented puzzles that require two or more sequential mental manipulations of the key to deriving a final solution. Increasingly difficult items involve a mix of puzzles that requires fluid shifts in deduction, logic, and inference.

 Wechsler Intelligence Scales for Children, Fourth Edition
The [Wechsler Intelligence Scales for Children, Fourth Edition (WISC-IV)]is used to have an overall measure in cognitive ability with five primary indexing scores. In the WISC-IV, the Perceptual Reasoning Index contains two subtests that assess _g_f: Matrix Reasoning, which involves induction and deduction, and Picture Concepts, which involves induction.
1) Picture Concepts: In the Picture Concepts task, children are presented with a series of pictures on two or three rows and asked which pictures (one from each row) belong together based on some common characteristic. This task assesses the child's ability to discover the underlying characteristic (e.g., rule, concept, trend, class membership) that governs a set of materials
2) Matrix Reasoning:- Matrix Reasoning also assesses this ability as well as the ability to start with stated rules, premises, or conditions and to engage in one or more steps to reach a solution to a novel problem (deduction). In the Matrix Reasoning test, children have presented with a series or sequence of pictures with one picture missing. Their task requires the child to choose the picture that fits the series or sequence from an array of five options. Since Matrix Reasoning and Picture Concepts involve the use of visual stimuli and do not require expressive language, they have been considered to be non-verbal tests of _g_f.


Unlike the Wechsler Scales, which assess a broad range of cognitive abilities including verbal comprehension, Raven's Progressive Matrices is a non-verbal intelligence test that primarily measures abstract reasoning and pattern recognition.[Link](https://www.psychologistmanjuantil.com/2024/12/types-of-intelligence-tests-wais-wisc.html)


Ref: [Bongard LOGO](https://dl.acm.org/doi/pdf/10.5555/3495724.3497106)   [directly inspired by the design principles behind the BPs]:
1) **Context dependent perception**: As in Indian kitchens, a dabba of a gym suppliment can be used to store masala, if asked What is stroed in this dabba, does VLM used the context to understand what is stored in this. or An equilateral triangle can either be interpreted as equilateral_triangle or as convex, depending on other contextual images.

2) **Analogy making perception** : Eclipse and a Man Blocking a Movie: A visual analogy can simplify the concept of an eclipse by comparing it to a person standing between the viewer and a movie screen. This helps understand how the moon blocks the sun's light during an eclipse.

3) **Perception with a few examples but infinite vocabulary**: Figure 1a shows a representative example of free-form problems, where we humans cannot provide the precise name of the free-form shape, as a concept, but we are still able to easily recognize the concepts by observing the strokes, and then make the right decisions on classifying novel test images


Ref: [V-PROM](file:///C:/Users/HP/Downloads/6885-Article%20Text-10114-1-10-20200525.pdf)
It focuses on real image but basde on the work of [Measuring abstract reasoning in neural networks | Barrette et. al. ](https://proceedings.mlr.press/v80/barrett18a/barrett18a.pdf) Uses visual Genome for dataset contruction.

Ref [Measuring abstract reasoning in neural networks | Barrette et. al. ](https://proceedings.mlr.press/v80/barrett18a/barrett18a.pdf)
It is dataset based on Ravens Progressive Matrix to text ml Models vision perception capability

Ref: [Bongard-HOI](https://openaccess.thecvf.com/content/CVPR2022/papers/Jiang_Bongard-HOI_Benchmarking_Few-Shot_Visual_Reasoning_for_Human-Object_Interactions_CVPR_2022_paper.pdf)
We introduce Bongard-HOI, a new visual reasoning benchmark that focuses on compositional learning of human-object interactions (HOIs) from natural images. It is inspired by two desirable characteristics from the classical Bongard problems (BPs): 1) few-shot concept learning, and 2) context-dependent reasoning. It studies human-object interactions (HOIs) as the visual concepts, requiring explicit compositional reasoning of object-level concepts. Our Bongard-HOI benchmark inherits two important characteristics of the classic Bongard problems (BPs)
1) **few-shot binary prediction**, where a visual concept needs to be induced from just six positive and six negative examples and 
2) **context-dependent reasoning**, where the label of an image may be interpreted differently under different contexts.

Ref [Do you see me](https://arxiv.org/pdf/2506.02022)
It has designed its visual perception tasks based on [CENTRAL PROCESSINGW DYSFUNCTIONS IN CHILDREN](https://files.eric.ed.gov/fulltext/ED040546.pdf) But ibelieved in this work we can extract more dense perception benchmark than do you see me.

Explore more real Bongard problems 
Use Wechsler scale to design broad scale problems for VLMs
Explore the idea of mirror reflections / self positions / reasoning through occlussions / temporal causality ( like image of a tilted glass will the model say what happned next or what has happened ) / can it understand confounding causality in vision ( Like Children with Bigger Shoes Have Better Reading Skills  **Observed Relationship (in image)**: A photo shows several children lined up with **larger shoe sizes** reading books fluently, and others with **smaller shoes** struggling to read. **Confounding Variable**: **Age**)
read this [Link](https://catalog.nlm.nih.gov/discovery/fulldisplay/alma996012263406676/01NLM_INST:01NLM_INST) ,[Link](https://pmc.ncbi.nlm.nih.gov/articles/PMC3204942/#S2) [Link](https://www.cell.com/iscience/fulltext/S2589-0042(21)00360-6?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS2589004221003606%3Fshowall%3Dtrue) [Link](https://oecs.mit.edu/pub/8w58nrk1/release/2) so that we caan come up with something like Do you see me
Test hypothesis with VLMEvalKit