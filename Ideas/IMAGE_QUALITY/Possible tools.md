The goal of this is to use various tools to assess the quality of image in a dataset. 
Search:
 - chatgpt [ prjt ] - has a thread
 - perplexity [ rittika bita ] - has 2 thread

1. Noisy image experiment : [Aditya] - make the image gray scale and see if an MLLM is sill answering the question
2. Minimal-pairs / contrast sets (compositionality & local edits). What it finds: Reliance on hackable surface cues; failure on small but meaning-changing edits. How: Build _contrast sets_ by minimally editing questions (negation, quantifier, relation flip) or options; track flips in ground truth vs predictions. Citations: Contrast sets (NLP general), Winoground (vision-language minimal pairs), SugarCrepe (de-biased compositionality).

- Qualitative CoT and answer from the existing evaluation of these benchmark.
- What kind of evaluation method is being used in these benchmark ( like string matching, etc ).
- Sample question from these benchmark and manually go through these question to find any error (Issue with questions framing, bias in the question ).
- List out the kind of issues we see ?
- 
- 