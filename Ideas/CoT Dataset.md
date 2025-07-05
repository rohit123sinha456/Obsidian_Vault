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


| Concepts ><br>Benchmark v<br> |     |     |     |     |     |     |     |     |
| ----------------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
|                               |     |     |     |     |     |     |     |     |
Ref: [Bongard LOGO](https://dl.acm.org/doi/pdf/10.5555/3495724.3497106)   [directly inspired by the design principles behind the BPs]:
**Context dependent perception**: As in Indian kitchens, a dabba of a gym suppliment can be used to store masala, if asked What is stroed in this dabba, does VLM used the context to understand what is stored in this. or An equilateral triangle can either be interpreted as equilateral_triangle or as convex, depending on other contextual images.

**Analogy making perception** : Eclipse and a Man Blocking a Movie: A visual analogy can simplify the concept of an eclipse by comparing it to a person standing between the viewer and a movie screen. This helps understand how the moon blocks the sun's light during an eclipse.

**Perception with a few examples but infinite vocabulary**: Figure 1a shows a representative example of free-form problems, where we humans cannot provide the precise name of the free-form shape, as a concept, but we are still able to easily recognize the concepts by observing the strokes, and then make the right decisions on classifying novel test images

Ref: [Bongard-HOI](https://openaccess.thecvf.com/content/CVPR2022/papers/Jiang_Bongard-HOI_Benchmarking_Few-Shot_Visual_Reasoning_for_Human-Object_Interactions_CVPR_2022_paper.pdf)
We introduce Bongard-HOI, a new visual reasoning benchmark that focuses on compositional learning of human-object interactions (HOIs) from natural images. It is inspired by two desirable characteristics from the classical Bongard problems (BPs): 1) few-shot concept learning, and 2) context-dependent reasoning