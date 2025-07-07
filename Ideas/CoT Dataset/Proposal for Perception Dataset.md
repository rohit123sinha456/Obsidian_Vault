Visual Memory (Sequential & Simultaneous)
- **Definition:** Recalling visual stimuli after brief exposure.
- **Potential Test:** Show models a frame → remove it → ask questions requiring recall of positions/colors/shapes.
- **Psych. Origin:** Visual span/memory tests like Corsi block-tapping or Rey-Osterrieth Complex Figure Test.
- PAM Dataset also talks about this

Illusory Perception & Visual Ambiguity
- **Example:** Necker cube, Rubin vase, or Shepard's tables.
- **Cognitive Angle:** What do illusions tell us about top-down processing?
- **MLLM Test:** Can models describe both interpretations? Or resolve perceptual conflicts?

Confounding concepts in Images
- Like bigger shoes means better reading ability
- I have to think more maybe idk

Recursive Structures Detection (Strange Loops) [[GEB Hoffstadter]]
- Task: Identify figures that contain **self-similar structures** (e.g., a spiral of spirals, a hand drawing a hand).
- Input A grid of images, some containing recursive motifs.- Show a sequence of nested images (e.g., a person drawing a picture of a person drawing a picture…).
- Ask: Who is drawing the final picture?” or “How many unique artists are in this recursive loop?” or  “Which of these images contains a structure that refers to itself?”
- Challenge Tests whether the model understands visual recursion,hierarchical loops visually and metaphor.




Visual Metaphor Matching
- **Task**: Match abstract shapes to metaphorical language (e.g., “This shape is like a conversation looping back on itself”).

Cross-Domain Isomorphism Detection
- Inspired by: GEB’s isomorphism between music, art, and math.
- Task: Given two different domains (e.g., a shape and a soundwave), identify **structural similarities**.
- Prompt “Which of these visual patterns has the same structure as this musical rhythm?”
- VLM Challenge Forces model to perceive beyond modality-specific cues.

Abstraction Trajectory Prediction
- **Task**: Given a sequence of progressively abstracted images (like sketch simplifications), predict the next or missing abstraction.
- **Idea**: Similar to Bongard patterns but emphasizing **semantic abstraction**, not just visual change.

Figure-Ground Ambiguity Resolution
- Inspired by Hofstadter’s interest in ambiguity and shifting perspectives.
- Task Present ambiguous figures (e.g., Rubin's vase) and ask VLMs to describe both interpretations
- Goal Test flexible perception and multi-perspective understanding.

Rule Inference with Symbolic Swaps
- Inspired by Bongard’s structural patterns & GEB’s symbol play.
- Task Given a set of positive and negative examples (à la Bongard problems), infer the transformation rule (e.g., "swap triangles and circles while preserving position").
- Model Prompt “What distinguishes the left set from the right?”
- Challenge Requires reasoning about abstract visual rules.

Chunking and Dechunking Tasks
- Task Given a complex pattern, identify which subpatterns are functionally grouped (chunking), or re-compose them (dechunking).
- Cognitive Test Can the model parse visual scenes like humans chunk words in language or notes in music?

