We use this [https://arxiv.org/pdf/2511.06209v2] idea of u heads. Using a u_head we classify upto kth step of reasoning whether the model needs to backtrack or reformulate the reasoning like this[https://openreview.net/pdf?id=OwhVWNOBcz]. The using some form of steering vector for reasoning or grounding we fornce the model to back track and reason again or

Lets say the uhead told that the reasoning step $k^{th}$ is not grounded in image. we remove that step from the reasoning, we feed the model $(I,Q,R^{<k-1})$. apply the steering vector or reasoning or grounding and then continue generation.


Problems needed to be solved.
1. What can be the type of data we can use to teach this UHead grounding (or any form of reasoning labels .)
2. Get the steering vector.
3. what kind of data do we need for the steering vector.
4. Does applyng steering vector in bettwen reasoning trace works ? If nor what can be the formulation to do that.
5. 
