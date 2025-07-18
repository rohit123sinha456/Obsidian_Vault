

##### Context-Based Perception
- An equilateral triangle can either be interpreted as equilateral_triangle or as convex, depending on other contextual images
##### Analogy Making Perception or [Conceptual slippage](https://sci-hub.se/https://doi.org/10.1007/978-3-642-34422-0_2)
- Shape analogy - A circle is to a sphere as a square is to a cube
- Functional analogy Spoon is to bowl as fork is to plate
- Spatial configuration Books on shelf is to office as plates on rack is to kitchen
- Color/form mapping Green leaves is to tree as red petals is to flower
- Object-role  Police car is to traffic as ambulance is to hospital
##### Visual Metaphor Comprehension [MetaCLUE](https://arxiv.org/pdf/2212.09898)(https://arxiv.org/abs/2305.14724)
-  Image of a ladder with people climbing labeled “career growth.”Ask: _“What does this image mean metaphorically?”_
##### Visual Memory ( Recalling visual stimuli after brief exposure )
- Like PAM
##### Confounding concepts in Images
- Can we do this ?
##### Recursive Structures Detection  (Strange Loops) [[GEB Hoffstadter]]
- a person drawing a picture of a person drawing a picture or like triangle fractals
- Here’s a curated list of datasets and resources that feature **recursive structures**, **visual metaphors for strange loops**, and concepts inspired by **Douglas Hofstadter’s** work (especially _I Am a Strange Loop_):


- **Datasets Featuring Recursive Structures & Strange Loops**

| Dataset Name                                       | Source                                                                                    | Description                                                                                                                         |
| -------------------------------------------------- | ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| **Fractal Image Dataset**                          | https://hirokatsukataoka16.github.io/Pretraining-without-Natural-Images/                  | Features computer-generated fractal images useful for exploring recursion and self-similarity.                                      |
| **Recursive Fractal Tree Dataset**                 | [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/recursive-fractal-tree-images) | Contains images of trees generated using recursive algorithms — strong metaphor for self-reference.                                 |
| **M.C. Escher Image Set (Unofficial)**             | [Flickr Group](https://www.flickr.com/groups/escher/pool/fdecomite/with/30528606454)      | Community-curated archive of Escher-like visuals. Not a formal dataset, but highly rich in recursive and paradoxical visual themes. |
| **DeepDream Images by Google**                     | [GitHub](https://github.com/google/deepdream)                                             | Dreamlike outputs that form visual feedback loops. Though neural, they strongly resemble strange loops in visual form.              |
| **The Mandelbrot Set Dataset**                     | [Kaggle](https://www.kaggle.com/datasets/yashsaxena/the-mandelbrot-set)                   | Mathematical fractals with infinite recursion; excellent for strange-loop-themed learning.                                          |
| **Recursive Neural Networks for Visual Reasoning** | [Stanford](https://cs.stanford.edu/people/karpathy/recnn/)                                | Includes structural visual data composed recursively; ties closely with Hofstadter’s computational loop ideas.                      |
| **Infinite Zoom Art Dataset**                      | [HuggingFace](https://huggingface.co/datasets/fusing/infinite-zoom-art)                   | Looping art forms with endless zoom — artistically conveys the recursive identity and paradox of strange loops.                     |


##### [Bistable perception](https://arxiv.org/html/2405.19423v1) or Figure Ground ambiguity or more genrally Perspective Shift (BiStable, Multistable etc) :
- like rubins vase

##### Perspective Taking
- Understanding what a scene looks like from another viewpoint (spatial or social) What can the person behind the box see? From this angle, which object is hidden? What would person A assume is happening here? 
- Image of two kids with a toy hidden from one. Q: Who knows where the toy is? ( Perspective type - Theory of Mind Based )
- Image beyond line of sight or occluded from line of sight. Q: Can this person see the ball? ( Line of Sight)
- Image showing a person facing a photoframe. Q: what image is visible to them ( Viewpoint-based perception)
- **Image 4**: Over-the-shoulder shot of a person looking at a painting.**Q**: Which image are they looking at?  **A**: "Starry Night" (Perspective-view linking).
- **Image 5**: A person facing a mirror, with an object behind them.**Q**: Can they see the object?  **A**: Yes, through the mirror (Indirect perception).
- **False Belief Reasoning**: Reasoning about scenarios where the agent holds a belief that contradicts reality (classic ToM test). Sally-Anne style tasks with image sequences and prompts
- Testing theory of mind in large language models and humans [Link]](https://www.nature.com/articles/s41562-024-01882-z.pdf)
##### Abstraction Trajectory Prediction ( Bongard Turns )
- so like a mix of identifying turn problems in bongard, Mind the gap path plans, or like spatial navigation. 
##### Ego centric and allocentric perception ( Frame of reference switching ):
- Can the model understand to your left or to your right. We can introduce the idea of visual fields ( cones of +- 20 degrees form the center ). So while constructing the dataset we can use this cone to say which objects are left to the "Self" and which objects are right. and later we can probe to see does the model learn this idea of cone for ego centric perception.
- We can cover the idea of left right and other allocentric perception as per VSR.
#####  Affordance Reasoning
- Spatial:  If I am standing near the table can i reach the cuboard.
- function: A scene with several objects (e.g., chair, ball, knife, cushion) Q: Which of these objects can be sat on? or Which object affords grasping with one hand?
- Ego Centric: A tall stool and a toddler next to it Q: Can the child sit on the stool unaided?
- Context-Sensitive: Rainy and sun dry image: Does the umbrella afford use in this setting?
- Perspective shifted: A book propping open a door, a shoe used to hammer a nail. like with an image of a nail, we ask what can be used to drive the nail n, Shoe, or glass bottle
- geometry and function: geometry and function Q; Which of these objects affords easy lifting? or Which surface affords sitting?
##### Spatio-temporal reasoning
- If the ball is roling from left to right. If i remove the 
##### Multi-rule compositional reasoning
##### Multi-step compositional reasoning

##### Free form answers
- Most datasets have like options to chose from which can lead to issues as mention by MMReason. For our task can we do like free form answers for evaluations

##### Visual isomorphism detection :
- Recognizing structural similarity between different domains or formats
- like A seesaw and a balance equation. and questions like Which of these images shares the same structure as the top image? or Identify the diagram that is isomorphic to the given network.
- Here are **10 text pairs**, each illustrating a **different aspect of isomorphism**, including structural, functional, relational, spatial, hierarchical, cyclical, symmetrical, dynamic, algebraic, and conceptual isomorphisms:

- **1. Structural Isomorphism**

	**A**: _A spider web with radial lines connected by concentric circles._  
	**B**: _A bicycle wheel with spokes radiating from the center to a circular rim._
	
	➡ Both have identical radial and circular structure despite different materials and contexts.

-  **2. Functional Isomorphism** [ its a bit difficult to generate image ]

	**A**: _A light switch turning on a bulb._  
	**B**: _A software button triggering a screen to light up._
	
	➡ Different mechanisms (physical vs. digital), same input-output relationship.

-  **3. Relational Isomorphism**

	**A**: _A family tree showing parent-child relationships._  
	**B**: _A file directory tree showing folder-subfolder structure._
	
	➡ Nodes and hierarchical relationships are preserved across domains.

- **4. Spatial Isomorphism**

	**A**: _A tic-tac-toe grid with Xs and Os in diagonal opposition._  
	**B**: _A warehouse layout with marked zones diagonally across the floor._
	
	➡ Relative spatial relationships are maintained.

-  **5. Hierarchical Isomorphism**

	**A**: _An organizational chart of a company with departments under a CEO._  
	**B**: _A food web with apex predators at the top and producers at the bottom._
	
	➡ Both represent layered levels of influence or flow.

-  **6. Cyclical Isomorphism**

	**A**: _The water cycle: evaporation → condensation → precipitation → collection._  
	**B**: _The software development cycle: plan → develop → test → deploy → repeat._
	
	➡ Different domains with an identical repeating loop structure.

-  **7. Symmetrical Isomorphism**

	**A**: _A butterfly’s wings mirrored on both sides of its body._  
	**B**: _A Gothic cathedral’s front facade with identical towers._
	
	➡ Mirror symmetry is preserved in both visual forms.

-  **8. Dynamic Isomorphism**

	**A**: _A pendulum swinging left and right around a central point._  
	**B**: _A political opinion poll swinging between two candidates over time._
	
	➡ Temporal fluctuation patterns are isomorphic, though one is physical and the other social.

-  **9. Algebraic Isomorphism**

	**A**: _Integers under addition._  
	**B**: _Even integers under addition, mapped by doubling._
	
	➡ There exists a bijective homomorphism (f(x) = 2x) preserving structure.

-  **10. Conceptual Isomorphism**

	**A**: _A courtroom where evidence is weighed on both sides before judgment._  
	**B**: _A balance scale weighing two substances for equilibrium._
	
	➡ Abstractly identical: both weigh competing entities to reach a decision.


##### Visual closure
- Like do you see me, instead of figures and we identify the salient region of an image for atask and then obscure certain part of the salient image to see if the task can be completed or not.
   
##### Perspective Reasoning
- Present ambiguous or flattened drawings and test the model's ability to reason about spatial depth and camera viewpoint. like if i give drawing of circle and ask looking from any other perspective what would the object look like cylinder, Qube, pyramid or sphere. answer, sphere and cylinder

##### Relative spatial orientation
- Like that captcha, that is the hand pointing in the direction of the car or something like that

##### Topological Reasoning:
- **Hole Counting Task** : Image shows several objects (e.g., donut, pretzel, mug, disk) and ask the question Which of these objects has the same number of holes as the mug?
- Knots and Untangling : Multiple string-like figures: some are knots, others can be untangled. question like Which of these knots can be untied without cutting?
- Enclosure and Inside-Outside Reasoning : Several shapes: some with one shape inside another, some touching, some outside. Que like Which figure shows a square inside a circle? Point to the shape that encloses the triangle.
- Topological Transformation Equivalence : One shape (e.g., circle), and multiple candidates: ellipse, twisted loop, figure-eight, triangle. Which shape can be deformed into the first shape without cutting or gluing?
- Maze continuity : A path/line figure with some being continuous loops, others broken or disconnected. Which path has a break? or Which path has a loop?

##### Dynamic manipulation of spatial structures mentally
- Mental Rotation : Imagining rotating a 2D or 3D object to a different orientation like in Mind the Gap and some custom task. Can the model predict what a shape will look like after a transformation (e.g. rotate in any axis)
- Mental Folding: Simulating folding of flat shapes along lines to visualize a resulting 3D shape. Like that T-Shaped paper when folded along the line create a cube or that diamond on a square paper when folded create a pyramid.
- Mental Unfolding (Unfolded View): Like when a cylinder is unfolded it looks like a rectangle. 
- Unseen Face Identification: in unfolded state if each figure has a number, can it say that is number x is at the front which number will be at the back of the fully folded structure
- Mental Paper folding aka Hole Displacement After Folding. given A flat sheet with one or more holes and fold lines. Question is Where will the hole appear after folding along the given lines?
##### Bongard Problems
- Can we use Bongard-LOGO and Bongard-HOI and get only the tough probelms after evaluating bu VLMs like MMReason
- Can we think in some ways to include bongard problems concepts in natural images in this dataset or like 




understand where the model is failing 
taxonomy of these tasks from visual perception to reasoning
feasibility of datasets create a table and say what can be created and what can be done
can we elicit the failue understanding of the models in this task
