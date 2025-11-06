With the visual grouding method of TRIQA we check if the traces are grounded or not.
With Aditya's method we can check if the reasoning line is perception heavy or reasoning heavy. If they are perception heavy they must have a high grounding score.
Based no that verifier paper that uses information theoritical thing where all the reasoning steps must be right at any point if its wrong henceforth all are wrong. based on that we can check for a reasoning heavy steps if they are dependednt ont he previous perception and reasoing step or not ( like thought anchors)

If all goes well then only the cot is verified.






For CoT Steps
Use MCTS with same model for each step to generate N rollouts to get the final answer from that step . That would give the "correctness of that step" + Use LLM-as-a-judge so same step annotation. Both must agree on each steps score for that step to be valid.

Identify visual step vs reasoning steps.
For each visual step 
1. MLLM-as-ajudge to check grounding using prompt.
2. Reframe that reasoning step as a question and give the model with image corrupsted ( gray noise or adversarial noise ) . if same conclusion is reached, step is not grounded else grounded.
3. Both must agree to give the score

Use the mapping mind of LLM, idea of graphs, to check if the reasoning converges a central answer and does that match with the final answer.

For evaluation
we can test BoN across some real world datasets.

