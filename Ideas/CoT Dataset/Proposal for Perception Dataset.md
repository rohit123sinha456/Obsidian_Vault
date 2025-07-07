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

Visual Escherian Paradoxes
**Concept**: Use impossible objects (e.g., Penrose stairs, Escher's “Relativity”).
**Task**:
- Ask: _“Where is the water flowing?”_ or _“Which direction is the person walking?”_
**Extension**:
- Modify classic Escher scenes with slight inconsistencies and test if the model can localize the paradox.
Purpose: Assesses spatial logic, global coherence, and whether the model detects physically impossible structures.


Visual Metaphor Comprehension
- **Concept**: Hofstadter loved metaphor and its role in cognition.
- **Task**: 
	- Match abstract shapes to metaphorical language (e.g., “This shape is like a conversation looping back on itself”).
	- Image of a ladder with people climbing labeled “career growth.”
	- Ask: _“What does this image mean metaphorically?”_
**Purpose**: Probes whether VLMs can understand visuals beyond literal semantics.

Visual Gödel Sentence
**Concept**: Encode self-referential logic in visual scenes.
**Task**:
- Present an image labeled: _“This image contains no labels.”_
- Ask: _“Is the statement true or false?”_
**Variation**: Use signage in a scene (e.g., _“All signs in this room are false”_).
**Purpose**: Forces the model to resolve logical paradoxes grounded in visual perception.


Cross-Domain Isomorphism Detection
- Inspired by: GEB’s isomorphism between music, art, and math.
- Task: Given two different domains (e.g., a shape and a soundwave), identify **structural similarities**.
- Prompt “Which of these visual patterns has the same structure as this musical rhythm?”
- VLM Challenge Forces model to perceive beyond modality-specific cues.

Abstraction Trajectory Prediction
- **Task**: Given a sequence of progressively abstracted images (like sketch simplifications), predict the next or missing abstraction.
- **Idea**: Similar to Bongard patterns but emphasizing **semantic abstraction**, not just visual change.

Figure-Ground Ambiguity Resolution and prespective shifts
- Inspired by Hofstadter’s interest in ambiguity and shifting perspectives.
- Task Present ambiguous figures (e.g., Rubin's vase) and ask VLMs to describe both interpretations
- Task: count the number of triangles in that triangular fractal
- Goal Test flexible perception and multi-perspective understanding.

Rule Inference with Symbolic Swaps
- Inspired by Bongard’s structural patterns & GEB’s symbol play.
- Task Given a set of positive and negative examples (à la Bongard problems), infer the transformation rule (e.g., "swap triangles and circles while preserving position").
- Model Prompt “What distinguishes the left set from the right?”
- Challenge Requires reasoning about abstract visual rules.

Chunking and Dechunking Tasks
- Task Given a complex pattern, identify which subpatterns are functionally grouped (chunking), or re-compose them (dechunking).
- Cognitive Test Can the model parse visual scenes like humans chunk words in language or notes in music?



[Concept Craft: a Poetics of AI Art](https://www.amazon.com/Concept-Craft-Poetics-AI-Art-ebook/dp/B0C6NXMKTX/?_encoding=UTF8&pd_rd_w=i8SFx&content-id=amzn1.sym.cf86ec3a-68a6-43e9-8115-04171136930a&pf_rd_p=cf86ec3a-68a6-43e9-8115-04171136930a&pf_rd_r=135-7710023-3695358&pd_rd_wg=BZd2r&pd_rd_r=65b50959-1319-4096-bd59-e2051db01fd1&ref_=aufs_ap_sc_dsk).

Conceptual slippage
Figure Ground ambiguity
Analogy based
Perception Shift
metaphor comprehension
isomorphism detection
perspective taking

