The goal of this is to use various tools to assess the quality of image in a dataset. 
Search:
 - chatgpt [ prjt ] - has a thread
 - perplexity [ rittika bita ] - has 2 thread

1. Noisy image experiment : [Aditya] - make the image gray scale and see if an MLLM is sill answering the question
2. Option Rotation: We can rotate the option to see if the position of the options effect the models performance [1].
3. Option order sensitivity for similar options : From the given options we can compute their similarity with the correct option, then we can check for the wrong answer how many time the more semantically similar option is chose. 
	1. We can create two sets: 
		1. Close distractors: replace the option with highest similarity with another option with more similarity, 
		2. Far distractors: replace all the options with options that have very low similarity with the correct answer.
	2. Then we can check the performance of the model in both the close distractor set and far distracttor set and see if distractors play any role
4. Option negation: We can negate the correct option[2] to see if the model still selects that option, so we can see if some kind of string matching happens or not.
5. Options Substitution : We can replace the option set of a IQA instance with the option set of another IQA instance to see if there is any memorization happening of QA pairs. If a model memorizes question - answer pairs, substituting the entire option set from other instances should break such memorization.
6. Give an image and 4 questions ( 1 true questions associated with that image and 3 questions from other instances). Ask the model to match which is the perfect question. see if the model can correct match the questions to the image ( check image to questions grounding )
7. Paraphrase the questions in a way that doesn't use that same keywords or same context words to see if that gives any changes in the models performance. This can test model's dependence on context keywords versus models grounding capabilities.
8. 




Papers
[1] Both Text and Images Leaked! A Systematic Analysis of Data Contamination in Multimodal LLM [https://arxiv.org/pdf/2411.03823]
[2] VQA-LOL: Visual Question Answering under the Lens of Logic [https://arxiv.org/pdf/2002.08325]

