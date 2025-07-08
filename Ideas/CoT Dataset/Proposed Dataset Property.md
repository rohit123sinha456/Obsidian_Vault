
##### Context-Based Perception
- An equilateral triangle can either be interpreted as equilateral_triangle or as convex, depending on other contextual images
##### Analogy Making Perception or [Conceptual slippage](https://sci-hub.se/https://doi.org/10.1007/978-3-642-34422-0_2)
- Shape analogy - A circle is to a sphere as a square is to a cube
- Functional analogy Spoon is to bowl as fork is to plate
- Spatial configuration Books on shelf is to office as plates on rack is to kitchen
- Color/form mapping Green leaves is to tree as red petals is to flower
- Object-role  Police car is to traffic as ambulance is to hospital
##### Visual Metaphor Comprehension
-  Image of a ladder with people climbing labeled “career growth.”Ask: _“What does this image mean metaphorically?”_
##### Visual Memory ( Recalling visual stimuli after brief exposure )
- Like PAM
##### Confounding concepts in Images
- Can we do this ?
##### Recursive Structures Detection  (Strange Loops) [[GEB Hoffstadter]]
- a person drawing a picture of a person drawing a picture or like triangle fractals

##### [Bistable perception](https://arxiv.org/html/2405.19423v1) or Figure Ground ambiguity or more genrally Perspective Shift (BiStable, Multistable etc) :
- like rubins vase

##### Perspective Taking
- Understanding what a scene looks like from another viewpoint (spatial or social) What can the person behind the box see? From this angle, which object is hidden? What would person A assume is happening here? 

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
