Create a CoT Based dataset for infographics Dataset

For domain specific Visual CoT Dataset, proposed framework.
Finetune/in-context learning of an LLM on that domain knowledge ( Like safety manual or other safety related documents ). Perform Open vocabulary object detection on the domain dataset, and then use LLM based CoT reasoning dataset for model finetuning/ alignment.

In Scene generation graphs like GAQ, we can take those questions that has multiple hops ( Like more than n hops ). We can select n like that fast and slow thinking paper did. we take n=1 hop perform inference see if the a model can answer proprly then we do n=2 hop and do the same and we n=k hop until k is the value where the model fails to give correct answer.

So in VoCot like dataset, For each reasoning step we can use an auxilarry LLM to check the textutal step and the crop of the image are grounded or not ( that is the textutal reason for that image crop is valid or not).  and for each step we can come up with like a grounding score. and at the end of reasoning we can check the cumulative grounding score to see if the reasoning is valid or not. 

Using a Critic CoT we can expand the VoCoT as such
{ "prompt": "...", "chain_of_thought": "...", "final_answer": "...", "grounded_steps": [true, true, false], // per-step grounding "coherence_score": 4, "faithfulness": "high" }


MMReason take a look
MME-CoT take a look
GFlowNet Take a look
Mulberry: Empowering MLLM with o1-like Reasoning and Reflection via Collective Monte Carlo Tree Search

Lets take the datasets from MME-CoT,MMMU, MMMUPro, MMStar, and M3CoT,VoCoT,NaturalBench,VisOnlyQA. 


During evaluation, we will only input open-ended textual questions while excluding the image to test whether the MLLMs can answer correctly without visual input. If the model can still answer correctly without any shortcuts or visual cues, we consider the instance potentially memorized or visually weakly relevant.

Most of these papers have MCQ based question or single answer based question. There is no Free form text answer.
 Grounding scoring: measure **IoU** or region descriptor accuracy against reference using grounding models
 Robustness Checks
 Introduce **visual distractor patches** or swap image parts
 Insert **adversarial region hints** in prompts; detect if models rely on these rather than true regions



Partial-CoT, given half the CoT steps can we regenerate the rest of the CoT correctly. If the models is goinf haywire it wont do that else it would.

Use Caption this read that paper dataset on VoCoT and check how good is its spatial reasoning task. What can we do for better spatial reasoing in CoT


CoT-Critic.

