Noah Seigel
1 What’s the difference between faithfulness and plausibility? In what situations might we care about each?

Faithfulness reflects the underlying reason for a decision. When taking a decision, if the reasoning steps to reach that decision is correct then that's faithful. On the other hand if the explanation or reasoning for a decision is sensible and convincing to a human observer ( that explanation or reasoning can be incorrect or deceitfull ) then that's plausible.

Fatihfulness has to be prioritized in high stakes decision making process like in healthcare ( The model has to be faithful when trying to identify any biomarkers for the doctors to rely on them ). Finance ( if using AI models for Bank loans or credit application/rejection, the customer would need a faithful answer and reason and not a plausible one)

Plausibility hold the priority when interacting with users in scenarios like product recommendation ( simply saying you might like "wrath of man"  because you watched "John Wick"  is enough instead of trying to explain a complicated recommendation system parameters used to suggest that movie ), or for simple communication where a broad range of people can get access to very niche and technical domains because of plausible explanations


1. Some faithfulness tests intervene on input examples (e.g. [https://arxiv.org/abs/2305.18029](https://arxiv.org/abs/2305.18029)); others intervene on model-generated CoTs (e.g. [https://arxiv.org/abs/2307.13702](https://arxiv.org/abs/2307.13702)). What information does each approach give us? What are the advantages of each, in terms of AI safety?

To draw up an analogy between these two approaches we can way interventions on input examples is like a detective trying to confirm a suspect's story by checking if the evidence at the scene matches their story, and Intervention on CoT is like the detective trying to cross examine the suspect and see if their story hold or not.

Intervention on the input examples tells us which specific words or data points actually drive the AI's final answer. Taking the example from the paper mentioned in the example[Table 1],  on inserting the word "blue" in the hypothesis, even though the prediction changed the world "blue" is not included in the Natural Language Explanation. From a safety perspective, this is incredibly important. It's how you discover if a loan-approval AI is actually looking at debt-to-income ratio, or if it's secretly (and illegally) basing its decision on the applicant's name or zip code. It helps us audit for hidden biases and ensure the AI isn't just taking shortcuts.

In the second line of work instead of changing the input, we mess with the "chain of thought." . What if we go into the reasoning and deliberately change something and see what happens is the primary RQ they are trying to answer. Lets take the CoT example from the example paper [Figure 1] 
"Step 1: 5!= 1x2x3x4x5. 
Step 2: 1x2x3x4x5 = 120.
Step 3: So the final answer is 120". 
When we stop the CoT at the first step the answer get is wrong. If we deliberately add a mistake in the CoT the final answer we get is wrong.
 it shows it's actually using its own reasoning. But if it corrects our flawed edit or, even worse, still gives the original answer of 120, it suggests the whole chain-of-thought was just for show. It likely knew the answer all along and just generated a plausible-sounding explanation afterward.

In terms of AI safety, this is about catching deception. We don't want powerful AI systems that can figure out an answer through some complex, black-box process and then generate a simple, believable story to hide their true methods. This approach helps us check if the AI's reasoning is just a performance. It's a way to ensure the transparency we think we're getting is real and not just a convenient fiction. Ultimately, both methods are crucial. The first tells us if the AI's reasoning is grounded in the right evidence, while the second tells us if the reasoning process itself is genuine. You need both to be truly confident that you understand what the AI is actually doing.

1. Consider the paper “Walk the Talk”: [https://arxiv.org/abs/2504.14150](https://arxiv.org/abs/2504.14150). Point out a limitation or unanswered question, and propose the simplest experiment you can think of which would address/answer this. Be concrete: what libraries, model(s), and data would you use? What would you expect the results might be (low confidence guesses are fine!), and how would you visualize these results in a plot?

