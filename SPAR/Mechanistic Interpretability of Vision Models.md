Proposal Form Mentor

## Proposal

Description of the project

I'm very open to working on directions that are of interest to my mentees. I find projects go much better when we can find common interests. That said, let me share what I'm particularly excited about right now.

  

I'm especially enthusiastic about vision model interpretability, with InceptionV1 being my favourite to study. Why vision models? They give us this unique window into phenomena that are just harder to study in language models. Feature manifolds, for instance, are so much more intuitive (curve detectors are a curved manifold)! Plus, InceptionV1 is tiny by today's standards (~7M parameters) which makes it perfect for independent projects where you want to actually understand the whole system rather than just poking at one corner of a massive model.

  

Some specific projects I'd love to explore in vision models:

- Understanding adversarial examples through the lens of interpretability. Can we use sparse autoencoders to decompose what's actually happening when an adversarial example fools a network? What circuits are being hijacked?
- Studying feature manifolds. E.g. identifying them, understanding their geometry, or looking at how they evolve through layers.
- Comprehensive circuit analysis. A model like InceptionV1 is quite straightforward (no attention mechanism or residual stream!) and so it seems quite likely that we could map out circuits end-to-end (albeit massive ones).

  

I'm also really interested in building tools and infrastructure that enable better interpretability research. This is not only important for the field but also fantastic for developing research skills. Some examples:

- Building better open-source tooling for interpretability research.
- Creating model organisms.
- Developing visualization tools that help us actually understand what we're looking at when we peer inside these models.

  

The engineering side is especially great if you're interested in having immediate, tangible impact while building up your research intuitions. Plus, good tools have a multiplier effect on everyone else's research!

Briefly, how does your project advance AI safety?

Doing interpretability on vision models lets us tackle questions that are much harder to reason about in LLMs (e.g. feature manifolds, feature symmetry) while developing techniques that should transfer to large frontier models. This creates a fast iteration loop for ambitious interpretability problems. The field is also at an inflection point: there's rapidly growing interest in interpretability research, but we're bottlenecked on accessible on-ramps for talented engineers. There are many engineering-heavy projects that could unlock real research progress, and I think investing in both these projects and the engineers who can execute them is crucial for scaling our ability to understand increasingly powerful AI systems.

What role will mentees play in this project?

Mentees will lead the day-to-day technical work of the project. The level of autonomy with direction is flexible and I’m happy to provide more or less guidance depending on the mentees’ preferences.

## Research Question
Question(s) for applicants

Without worrying too much about feasibility, if you had unlimited compute and 6 months, what interpretability question would you want to answer? (250 words)

A: Without worrying about feasibility and compute I would try to understand the geometric view of adversarial vulnerabilities. So the goal would be to create an "Atlas" of the the feature space. So I would trace the activation region of every feature to create its manifold. From this manifold, we can use the literature of riemannian manifold to understand the curvature and topology of this manifold. So this is the feature manifold space of the model. The we can use any technique like PGD ( adversarial optimization process ) and trace its trajectory from correct example to adversarial example across this geometric space . This would allow us to identify what can be bottleneck of these vision models from a geometric perspective. This create a differential geometry based intuition which is transferable to the abstract conceptual geometries of far larger models
  
The idea of using riemannian geometry for mapping out quotient spaces in a manifold to understand model brittleness comes from this work [Hide & Seek: Transformer Symmetries Obscure Sharpness & Riemannian Geometry Finds It ]


What's one thing you think the mechanistic interpretability community is getting wrong or overlooking right now? This could be a methodological issue, a common assumption, or a research direction. Explain your reasoning. (250 words)

This work [RECIPROCAL LEARNING: https://arxiv.org/abs/2408.06257] posits that learning parameters from data is in fact a reciprocal learning. In today's mechanistic interpretability landscape the work focuses on studying the final static artifact of the model but is largely overlooking the training process. Intuitively, we are trying to understand by reverse engineering a vase, how the vase looks, what is it made of, what id the quality of the ceramic the vase is made of etc, but we are not focusing on how the vase was made ( depending on whether its artisanal or commercial the quality would differ ), the temperatures used to bake the vase would determine its flexibility and etc. Just by looking at the fixed weights we are making the assumption that the learned algorithm of the model is a property of its weights and architecture alone. We are not looking into the implicit forces exerted on the model while training like the ordering of the training data ( cirriculum learning uses this property to drive model learning stability ), learning rate schedular, implicit bias of optimizer ( like SGD has a preference for flat minima ). This leads to a critical gap in our understanding. We might identify a circuit that understand the concept of synnetry, but we don't now why that particular circuit was formed. To simply put in words of Richard Fynman "What I cannot create, I do not understand" is an elegant way to put the works of circuits.

A more complete direction would be to involve tracing during circuit development (training) that would allow us to observe the emergence, competition, and stabilization of circuits.