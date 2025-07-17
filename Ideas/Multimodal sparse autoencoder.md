[SAE-V](https://arxiv.org/pdf/2502.17514) - It does three things
- Train the sparse autoencoder on the decoder feature dictonary - feature activation
- using cosine simalrity found out the text and vision features of this SAE for cross modal weighting
- then used this score to filter out data that has more cross modal weights so that the images are better aligned with the texts.

Proposal.
- they are extracting the join embeddings from VLM then they are training the sparse autoencoders
- we emphasizes **separately learning sparse feature dictionaries for image and text**, then aligning them into a **joint multimodal feature dictionary**.This allows for **cross-modal traceability**:
	 - Which **visual concepts** correspond to which **linguistic concepts**.
    - Explicit **alignment of latent concepts** across modalities.
    - SAE-V focuses on fused joint embeddings, but doesn’t separate or align modality-specific dictionaries explicitly.

- with this build a **symbolic interface** into the MLLM by using sparse features as:
	- **Named semantic variables**
	- That can be turned on/off
	- And composed into symbolic reasoning chains or **latent CoT paths**
	- SAE-V shows individual concept editing but does not frame this as a **symbolic control system** over MLLMs.

- **Use in Dataset Analysis and Alignment Auditing**
This approach suggests using sparse features to:
- Analyze **dataset biases**
- Identify **conceptual coverage gaps**
- Evaluate **alignment safety** at the level of abstract latent features
- SAE-V focuses on interpretability, not **auditing or alignment policy** based on feature activity statistics.

**Feature Naming via Retrieval and CLIP/BLIP**
You propose **automated naming of latent features** by:
- Retrieving top activators
- Using CLIP or BLIP to summarize what the feature captures
- SAE-V does manual interpretation of feature meaning.


Maybe
**Topology-Informed Interpretability (Fibre Bundle Perspective)**
You propose a **geometric interpretation**:
- Treat the sparse feature activations as **fibres over image-text input space**
- Interpret generalization and misalignment as **geometric misalignment of fibres**
- SAE-V doesn’t frame the problem with any topological structure or continuity guarantees.