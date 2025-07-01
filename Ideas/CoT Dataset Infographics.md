Create a CoT Based dataset for infographics Dataset

For domain specific Visual CoT Dataset, proposed framework.
Finetune/in-context learning of an LLM on that domain knowledge ( Like safety manual or other safety related documents ). Perform Open vocabulary object detection on the domain dataset, and then use LLM based CoT reasoning dataset for model finetuning/ alignment.

In Scene generation graphs like GAQ, we can take those questions that has multiple hops ( Like more than n hops ). We can select n like that fast and slow thinking paper did. we take n=1 hop perform inference see if the a model can answer proprly then we do n=2 hop and do the same and we n=k hop until k is the value where the model fails to give correct answer.


