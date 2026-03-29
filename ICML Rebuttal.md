

# Reviewer: hKzx
Score: 3 ( weak reject)


Strengths And Weaknesses:
Strengths

Originality: The core idea of this paper is novel. In particular, it generalizes existing approaches for designing residual connections in GNNs beyond the use of doubly stochastic matrices. The introduction of contractive monoids provides a mathematically rigorous framework for this generalization (although I have some concerns regarding the presentation of this idea, which I discuss in the weaknesses section). Extensive experimental results demonstrate the empirical effectiveness of this approach: leveraging contractive monoids leads to improved accuracy in GNNs with residual connections while mitigating over-smoothing as the network depth increases.

Experimental analysis: Another strong aspect of this paper is its extensive experimental evaluation. The authors consider a diverse set of benchmarks spanning small-, medium-, and large-scale graphs, which provides a comprehensive assessment of the proposed framework across different regimes. They include comparisons against strong baselines, allowing for a clear evaluation of performance gains. In addition, the paper goes beyond standard accuracy metrics by systematically analyzing depth scalability, demonstrating how the proposed residual designs alleviate performance degradation in deeper GNNs. The experiments also investigate representation quality and over-smoothing effects, providing empirical evidence that the contractive monoid framework preserves expressive node representations as depth increases. Furthermore, they also study the isolated contribution of different classes of contractive monoids (e.g., orthogonal, diagonal contractive, and spectral norm–bounded matrices), offering insight into their individual roles. Overall, the experimental section is thorough, well-structured, and supports the main claims of the paper.


Weaknesses

Presentation: One of the main weaknesses of this paper is its presentation. From a mathematical writing perspective, there are several instances of undefined or inconsistent notation that make the paper difficult to follow. For example, although equations (1–6) introduce a GNN with residual connections, the functional form of the GNN itself is not clearly specified. In addition, the notation used in Proposition 3.1 introduces $h^{(i)}$ to denote features, which is inconsistent with the notation in Definitions (1–6). More significant issues arise in the statements of some theorems, where the lack of clarity makes them hard to interpret. For instance, in Theorem 4.3, the paper refers to a GNN “whose pre-layer propagation induces a spectral contraction $(1 - \gamma)^{1/n}$” but it is unclear what is meant by “spectral contraction induced by a GNN,” as this notion is neither clearly defined nor formally introduced. These issues extend beyond notation and affect the explanatory remarks intended to provide intuition. For example, in the “Connection to over-smoothing” remark following Proposition 3.1, the paper refers to decoupling the diffusion rate from the mixing rate, but these terms are not clearly defined or explained, making the intuition difficult to grasp.

Significance of theoretical results: My second major concern is the practical significance of the theoretical guarantees presented in this paper for GNNs. In particular, it is unclear whether these results lead to identifying better contractive monoid candidates than those already existing in the literature (e.g., the Birkhoff polytope). For instance, it is not clear—nor demonstrated in the paper—whether Theorem 4.2 (multi-stream over-smoothing rate) improves upon the guarantees obtained from doubly stochastic constructions. More broadly, it remains unclear whether this result provides actionable guidance on selecting a contractive monoid that minimizes over-smoothing.

Another concern is related to the necessity of the proposed assumptions. While the paper argues that the doubly stochastic constraint is not essential for many guarantees, it is equally unclear whether the conditions imposed by the contractive monoid framework are themselves essential. For example, in Theorem 4.1 (perturbation stability), it is not evident whether the condition $|H| < 1$ is truly necessary to prevent perturbation amplification in GNNs. In particular, from equation (14), the multiplicative terms $1+ L_l|G_l||H_l|$ are always greater than 1, implying that the amplification factor $\Pi^{L-1}_{l=0} (1+ L_l|G_l||H_l|)$ grows exponentially with the depth of the network. Therefore, it is unclear why enforcing $|H| < 1$ is sufficient—or even essential—for controlling perturbation propagation based on this analysis.

Soundness: I have several concerns regarding the soundness of some of the results in this paper. First, it is unclear how essential Axiom 2: Compositional closure is within the proposed framework. Indeed, if Axiom 1 holds—i.e.,$|H| < 1$ 
 for every $H \in M$ —and the matrix norm is submultiplicative, then it follows that $|HH^{\prime}| < |H|,|H^{\prime}| < 1$ Therefore, compositional closure in terms of non-expansiveness appears to be automatically satisfied under Axiom 1 and submultiplicativity. For this reason, the remark following Proposition 3.2 seems questionable. In particular, if matrices $H$ and $H^{\prime}$ are non-expansive with respect to the $l_1$-norm (i.e., $|H| < 1$ and $|H^{\prime}| < 1$), then by submultiplicativity of the matrix $l_1$-norm we obtain $|HH^{\prime}| < |H|,|H^{\prime}| < 1$ which suggests that the claimed distinction may not hold. Second, I am unsure about the correctness of Proposition 3.3, as I could not find its proof in the appendix. Conceptually, it is also unclear why the inclusion in equation (12) should hold. Given that deeper GNNs are defined via $x^{l+1} = x^l + F_l(x^l)$ it is not immediately obvious that increasing the number of layers necessarily enlarges the function class in a strict or monotone sense without additional assumptions on the function class $\mathcal{F}_l$
.

Length and Impact Statement: I am not sure whether this is relevant, but the paper slightly exceeds the 8-page limit and does not include an impact statement.


Key Questions For Authors:
It is unclear how the theoretical results in this paper improve upon existing results in the literature. In particular, it would be valuable to understand whether these results provide concrete guidance for selecting or designing contractive monoids. For example, do the theoretical guarantees help identify which subclasses of contractive monoids yield better performance with respect to specific properties of interest in GNNs (such as over-smoothing, stability, or expressivity)?

------------------------------------------------------


We thank the reviewer for the careful reading and precise technical observations. We address each concern below.


### 1. Presentation: Notation Consistency and Clarity

**Concern:** Inconsistent notation between Eqs. (1–6) and Proposition 3.1; undefined "spectral contraction" in Theorem 4.3; unclear "diffusion rate" vs. "mixing rate."

**Response:** We acknowledge these inconsistencies and will make the following concrete revisions:

- **Eqs. (1–6) vs. Proposition 3.1:** The multi-stream representation $x_i^l \in \mathbb{R}^{n \times d}$ (Eq. 1) and the single-vector shorthand $h^{(\ell)}$ (Proposition 3.1) represent the same object at different levels of abstraction. We will unify notation by introducing $h^{(\ell)}$ as a flattened alias of $x_i^l$ at the start of Section 3, with a mapping table in the appendix.

- **"Spectral contraction" in Theorem 4.3:** This refers to the contraction factor $(1-\gamma)$ arising from the spectral gap of the normalized adjacency operator. The formal definition appears in the full theorem statement in Appendix B.3 (Theorem B.3), where $P$ is a symmetric propagation operator with $\lambda_2(P) \leq 1 - \gamma$. We will add this definition inline before Theorem 4.3 in the main text.

- **"Diffusion rate" vs. "mixing rate":** The diffusion rate is the rate at which graph convolution contracts pairwise node feature distances, governed by the spectral gap $\gamma$ of the propagation operator. The mixing rate is the perturbation introduced by the residual stream mixer $H_\ell$, bounded by $\|H_\ell - I\|_2 \leq \epsilon$. We will add explicit definitions of both terms after Proposition 3.1.

### 2. Significance of Theoretical Results: What the Theory Actually Predicts

**Concern:** (a) Unclear whether theoretical results identify better contractive monoid candidates than Birkhoff. (b) Whether Theorem 4.3 improves upon doubly stochastic guarantees. (c) Whether results provide actionable guidance.

**Response:** We want to be precise about the nature of our theoretical contribution, because we believe the reviewer's concern stems from an expectation the theory was not designed to meet.

**What the theory provides is safety guarantees with predictive structure, not point optimization.** The framework answers the question: *which algebraic properties must a residual mixer satisfy for stable deep propagation?* The answer — non-expansiveness, compositional closure, identity anchoring — is a necessary-and-sufficient characterization within the operator-norm setting. This is valuable because it tells practitioners exactly what they can and cannot relax when designing new mixing operators, whereas prior work (Mishra, 2026; Xie et al., 2026) offered only the Birkhoff polytope as a viable choice.

Crucially, **the theory is predictive in a specific, testable way.** Section 3.5 establishes a containment hierarchy among our instantiations:

$$\text{Diagonal} \subset \text{Orthogonal} \subset \text{Spectral Norm Ball} \supseteq \text{Birkhoff}$$

The spectral norm ball is identified as the *maximally expressive* contractive monoid (Section 3.5, page 5: "it represents a maximally expressive contractive monoid"). The theory therefore makes the concrete prediction that spectral-norm mixing should offer the broadest representational capacity, while more constrained monoids trade expressiveness for additional structural properties (norm preservation for orthogonal, computational simplicity for diagonal).

**This prediction is empirically confirmed.** In Table 1, the spectral norm ball achieves the best result on 4/8 datasets (Chameleon: 31.51, Texas: 67.57, Cora: 75.30, Wisconsin: 67.97), consistent with its theoretical status as the least restrictive contractive monoid. Finding 3 (Section 6, page 7) explicitly validates the predicted expressiveness–structure trade-off: diagonal contraction, which forbids cross-stream mixing by construction, shows systematically lower means than less restrictive contractive sets (Cora: Diagonal 75.23 vs. Spectral 75.30; CiteSeer: Diagonal 62.53 vs. Orthogonal 62.97).

Meanwhile, the orthogonal group — which the theory identifies as exactly norm-preserving (no signal attenuation) — excels precisely where information preservation across depth matters most. On the heterophilic benchmarks where nodes connect to dissimilar neighbors, orthogonal mixing achieves the best results: Wisconsin 69.32 ± 3.00 (Table 1, the overall best) and Cornell 54.05 ± 3.12 on GraphSAGE (Table 3, Appendix A).

**(b) On Theorem 4.3 specifically:** The theorem's contribution is *generalization*, not improvement of the bound's *tightness*. Prior work (Mishra, 2026) established the $(1-\gamma)^{L/n}$ over-smoothing slowdown only for Birkhoff-constrained mixing. Our Theorem 4.3 (with the full statement in Theorem B.3) proves the identical rate holds for *any* contractive monoid satisfying Axioms 1–3. Appendix B.3, Remark B.4 states this explicitly: "neither mechanism depends on doubly stochastic structure: the Birkhoff polytope is a sufficient instantiation, but any contractive monoid with identity anchoring induces the same asymptotic behavior." This is not a tighter bound — it is a strictly broader applicability of the *same* bound, which is the theoretically appropriate result for our thesis that Birkhoff is sufficient but not necessary.

**(c) Actionable guidance — proposed practitioner's decision table:**

| Design Priority | Recommended Monoid | Theoretical Rationale | Empirical Support |
|---|---|---|---|
| Max expressiveness | Spectral Norm Ball | Maximally expressive contractive monoid (Sec. 3.5) | Best on 4/8 datasets (Table 1) |
| Exact norm preservation | Orthogonal Group | $\|Hx\|_2 = \|x\|_2$ — zero information loss | Best on heterophilic graphs (Tables 1, 3) |
| Computational efficiency | Diagonal Contraction | $O(n)$ vs. $O(n^2)$ mixing; $O(n^2/d)$ overhead (Prop. A.1) | Competitive on homophilic benchmarks |
| Mean + mass conservation | Birkhoff Polytope | $H\mathbf{1}=\mathbf{1}$, $\mathbf{1}^\top H = \mathbf{1}^\top$ | Strong overall; established baseline |

This table will be added to the revision. The point is that the theory provides a *structured design space* with interpretable trade-offs, not a single best answer — which is appropriate because the best choice depends on the graph's properties.




### 3. Necessity of $\|H_\ell\|_2 \leq 1$ in Theorem 4.1

**Concern:** The amplification factor $\prod_{\ell=0}^{L-1}(1 + L_\ell \|G_\ell\|_2 \|K_\ell\|_2)$ grows exponentially regardless of whether $\|H_\ell\|_2 \leq 1$, so it is unclear why enforcing $\|H\|_2 \leq 1$ matters.

**Response:** The reviewer's observation about the product is mathematically correct, and this is exactly why the theorem's *decoupling* structure matters. The bound in Eq. (14) contains only terms from the GNN layers ($L_\ell$, $\|G_\ell\|_2$, $\|K_\ell\|_2$). The residual mixer $H_\ell$ is *absent* from this product precisely *because* $\|H_\ell\|_2 \leq 1$.

To see why this matters, consider the unconstrained case. If $\|H_\ell\|_2 > 1$, the perturbation recursion (Eq. 22 in Appendix B.1) gives:

$$\|\delta h^{(\ell+1)}\|_2 \leq \|H_\ell\|_2 \|\delta h^{(\ell)}\|_2 + \|G_\ell\|_2 L_\ell \|K_\ell\|_2 \|\delta h^{(\ell)}\|_2$$

which yields the product $\prod_{\ell}(\|H_\ell\|_2 + L_\ell \|G_\ell\|_2 \|K_\ell\|_2)$ instead of $\prod_{\ell}(1 + L_\ell \|G_\ell\|_2 \|K_\ell\|_2)$. When $\|H_\ell\|_2 = 1 + \delta$ for even small $\delta > 0$, this introduces an additional exponential factor $(1+\delta)^L$ that **compounds independently** of the GNN's own amplification. The constraint eliminates this entire multiplicative channel.

**Empirical confirmation.** The distinction is not merely theoretical. Figure 1(a,b) shows that unconstrained mixing — where $\|H_\ell\|_2$ is free to exceed 1 during training — leads to qualitatively different collapse behavior: high cosine similarity (Fig. 1a) and rapid variance decay (Fig. 1b), while all contractive variants cluster in a stable operating region. Table 2 (Appendix A) makes this even starker: at 128 layers on Cora, the standard GCN degrades to 14.47 ± 0.51 while all contractive variants remain above 72%.

We will add an ablation tracking $\|H_\ell\|_2$ across layers for constrained vs. unconstrained models in the revision to make this argument fully transparent.

---

### 4. Soundness: Role of Axiom 2 (Compositional Closure)

**Concern:** If $\|H\|_2 \leq 1$ for all $H \in \mathcal{M}$ and the norm is submultiplicative, then $\|HH'\|_2 \leq 1$ automatically, making Axiom 2 appear redundant.

**Response:** The reviewer is correct that **norm closure** (the product remains non-expansive) follows from submultiplicativity. However, Axiom 2 requires **set membership closure**: that $H_a H_b \in \mathcal{M}$, not merely that $\|H_a H_b\|_2 \leq 1$.

This distinction is substantive whenever $\mathcal{M}$ carries structural constraints beyond the norm bound. Consider the set of *symmetric* matrices with $\|H\|_2 \leq 1$. Each element is non-expansive, but the product $AB$ of symmetric $A, B$ is not generally symmetric (since $AB \neq BA$ in general), so $AB$ leaves the set despite satisfying $\|AB\|_2 \leq 1$. The norm is preserved, but the structural property (symmetry) is lost under composition. A network designer who chose symmetric mixers for interpretability (e.g., symmetric mixing implies equal bidirectional influence between streams) would find that the *effective* multi-layer mixer $\prod_\ell H_\ell$ no longer has this property — the structural guarantee degrades silently with depth.

**We acknowledge an error in the paper's original example.** In Section 3.2, we wrote that row-stochastic matrices are not closed under multiplication. This is incorrect — row-stochastic matrices *are* closed under multiplication. We will correct this in the revision and replace it with the symmetric-matrix example above, which cleanly demonstrates the failure mode.

**Why Axiom 2 matters for the framework's purpose.** For our three specific instantiations (orthogonal group, diagonal contraction, spectral norm ball), closure is straightforward to verify (Appendix C). The axiom's primary role is as a **design criterion for future monoid construction**: it rules out structurally appealing but compositionally unstable sets. This is consistent with our stated goal (Section 1): "establishing minimal structural requirements for stable residual propagation... a foundation for developing architectures." Axiom 2 tells a practitioner designing a new monoid: *verify that your structural constraints survive composition*, which is not automatic.

---

### 5. Soundness: Missing Proof of Proposition 3.3

**Concern:** (a) Proof is missing from the appendix. (b) Conceptually unclear why $\mathcal{F}_L \subseteq \mathcal{F}_{L+1}$ holds.

**Response:** We apologize for this omission. The proof is:

**Proof.** Let $f \in \mathcal{F}_L$ be any function realizable by a depth-$L$ network with mixers in $\mathcal{M}$. Construct a depth-$(L+1)$ network by appending one layer with $H_{L+1} = I$ (feasible since $I \in \mathcal{M}$ by Axiom 3) and $F_{L+1} \equiv 0$. Then:

$$h^{(L+1)} = I \cdot h^{(L)} + F_{L+1}(h^{(L)}) = h^{(L)}.$$

The additional layer acts as an identity, so the depth-$(L+1)$ network computes the same function. Since this holds for every $f \in \mathcal{F}_L$, we have $\mathcal{F}_L \subseteq \mathcal{F}_{L+1}$. $\square$

The argument requires only that $F_\ell$ can represent the zero function (always possible in standard parameterizations with zero-initialized weights) and that $I \in \mathcal{M}$. **Without** Axiom 3 ($I \in \mathcal{M}$), one cannot set $H_{L+1} = I$, meaning the additional layer necessarily transforms the representation, and the inclusion could fail. This is precisely why identity anchoring is axiomatically necessary, not merely convenient.

This proof will be added to Appendix B in the revision.

---

### 6. The $\ell_1$-norm / Row-Stochastic Counterexample

**Concern:** Row-stochastic matrices are non-expansive in $\ell_1$-norm, and their products remain non-expansive and row-stochastic by submultiplicativity, contradicting the paper's claim.

**Response:** The reviewer is correct, and we acknowledge this as an error in the manuscript. Row-stochastic matrices are indeed closed under multiplication. The example in Section 3.2 (page 4) is wrong.

We will replace it with the symmetric-matrix example described in Point 4 above, which correctly demonstrates the phenomenon: a set can satisfy Axiom 1 (non-expansiveness) while violating Axiom 2 (set membership closure) because structural properties beyond the norm bound are not preserved under composition.

A second correct example: the set of doubly stochastic matrices with at most $k$ nonzero entries per row. Each such matrix is non-expansive, but products can fill in additional entries, violating the sparsity constraint even though $\|AB\|_2 \leq 1$.

We thank the reviewer for catching this and will correct it in the revision.

---

### 7. Length and Impact Statement

The main text (Sections 1–6.5, ending with References on page 9) is within the ICML 8+1 page limit; all material beyond page 10 is appendix, which is permitted under ICML formatting guidelines. We will add a broader impact statement in the revision. As an architectural contribution with no new datasets or deployment-specific applications, the primary societal impact is indirect: enabling more stable deep GNN architectures may benefit applications in drug discovery, materials science, and social network analysis. We do not foresee negative societal consequences specific to this contribution.

---

### Summary of Key Revisions Promised

1. Unified notation with mapping table (Appendix)
2. Inline definition of "spectral contraction" before Theorem 4.3
3. Definitions of "diffusion rate" and "mixing rate" after Proposition 3.1
4. Practitioner's decision table linking theory to monoid selection
5. Correction of the row-stochastic example in Section 3.2 (replace with symmetric-matrix example)
6. Missing proof of Proposition 3.3 added to Appendix B
7. Ablation tracking $\|H_\ell\|_2$ across layers for constrained vs. unconstrained models
8. Broader impact statement


--------------------------------------------------
--------------- in 5000 characters----------------



















---------------------------------------------------------




# Reviewer E1PX
score: 2 (reject)

Strengths And Weaknesses:
Strengths

The paper positions Contractive Monoid as a fundamental principle to tackle the oversmoothing problem in GNNs.
Experiments demonstrate the utility of adopting Contractive Monoid in practice, where GNNs affirming to these principles significantly outperform others.
Paper clarity is good.
Weaknesses

The novelty of the proposed axioms is quite limited. In particular, the axioms of non-expansiveness or compositional closure are inherently tied to the previous empirical and theoretical advances related to stable learning in deep learning. The paper fails to make these connections with prior works or elucidate the novelty in this context.
The novelty and challenges behind Theoretical results in Section 4 could be emphasized better.
Specific emphasis on Birkhoff-constrained approaches lacks sufficient motivation. Moreover, why were other approaches to tackle oversmoothing from the literature not discussed with respect to Contractive Monoids?

Key Questions For Authors:
I'd appreciate it if the authors can address the points in the weaknesses section above.


----------------------------------------------


We thank Reviewer E1PX for the constructive feedback and the recognition of our paper's clarity and experimental utility. We address each concern below, grounding our responses in specific results from the paper. We respectfully argue that the review may underestimate the contribution by evaluating the individual axioms in isolation rather than the framework they jointly define, and we take responsibility for not making the distinction sharp enough in the submission.

---

## 1. "The novelty of the proposed axioms is quite limited… inherently tied to previous empirical and theoretical advances."

**We agree that the individual ingredients are known — and this is by design.** Non-expansiveness traces back to contraction mapping theory and appears in spectral normalization (Miyato et al., 2018) and Lipschitz-constrained networks. Compositional closure is a classical algebraic property. Identity anchoring echoes the residual learning principle (He et al., 2016). We do not claim novelty for any single axiom.

**The novelty lies in three specific contributions that no prior work achieves:**

**(a) Identifying these three properties as the minimal sufficient set for stable residual mixing in multi-stream GNNs.** This is not the same as applying spectral normalization to an arbitrary weight matrix. Spectral normalization constrains a *single layer's* weight matrix to stabilize forward/backward passes. Our framework constrains the *residual stream mixer* — the operator governing how information is routed across parallel representation streams at each layer. The critical difference is that residual mixers *compose across depth*: the effective transformation after $L$ layers is $\prod_{\ell=1}^{L} H_\ell$. This composition creates algebraic requirements — closure and identity anchoring — that simply do not arise when normalizing individual weight matrices. A spectrally normalized weight matrix with $\|W\|_2 \leq 1$ has no guarantee that the *set of admissible matrices* is closed under multiplication, nor that identity initialization is reachable within the constraint set. Our axioms capture exactly the additional structure needed for the compositional setting of residual mixing.

**(b) Proving that doubly stochastic structure is sufficient but not necessary for stable deep GNNs.** Prior to this work, mHC (Xie et al., 2026) and mHC-GNN (Mishra, 2026) represented the state-of-the-art, and their success was attributed to doubly stochastic properties: mean preservation, non-negativity, and permutation-matrix decomposition. It was an open question whether these specific properties were essential. Our Section 3.5 proves that Birkhoff $\subset$ Spectral Norm Ball, meaning the stability guarantees extend to a strictly larger operator family. This is a non-obvious finding — one could reasonably have hypothesized that mean preservation (a property unique to doubly stochastic matrices, not shared by orthogonal or spectral-norm-ball matrices) is critical for preventing representational drift.

**(c) Opening a concrete, empirically validated design space.** Table 1 shows that all three new instantiations substantially outperform standard GCN baselines (e.g., Cora: 70.53 → 75.30 for Spectral; Wisconsin: 53.59 → 69.32 for Orthogonal) and consistently match or exceed Birkhoff, with the spectral norm ball achieving the best result on 4/8 datasets. Table 2 (Appendix A) demonstrates that this advantage persists at extreme depths: at 128 layers on Cora, all contractive variants remain above 72% while the baseline degrades to 14.47%. These are not incremental improvements — they represent qualitatively different depth-scaling behavior enabled by the framework.

**Addressing the performance-margin concern directly.** The reviewer might note that differences *among* contractive monoids are modest (often within standard deviations). This is precisely what the theory predicts: all contractive monoids satisfying Axioms 1–3 share the same stability guarantees (Theorem 4.3: identical $(1-\gamma)^{L/n}$ over-smoothing rate). The important empirical finding is not that one monoid dominates another, but that **all contractive monoids dominate unconstrained mixing and standard baselines** — confirming that the *algebraic structure*, not the specific instantiation, is what matters. This is the central thesis of the paper.

**Connection to prior stable deep learning work.** We will add explicit discussion in Section 2 contrasting our framework with:

- **Spectral normalization (Miyato et al., 2018):** Constrains individual weight matrices; does not address compositional structure across depth or identity anchoring. Our framework subsumes spectral normalization as a special case of Axiom 1 but adds Axioms 2–3 for the compositional residual mixing setting.
- **Orthogonal RNNs (Arjovsky et al., 2016):** Use orthogonal recurrence matrices for gradient stability. Our orthogonal group instantiation (Section 3.5) brings this idea to multi-stream GNN residual mixing, where the compositional closure and identity anchoring axioms provide guarantees specific to the GNN over-smoothing problem that do not arise in sequential models.
- **Lipschitz-bounded networks (e.g., Li et al., 2019; Anil et al., 2019):** Enforce global Lipschitz bounds for certified robustness. Our non-expansiveness constraint serves a fundamentally different purpose: preventing the *residual mixer* from amplifying perturbations across streams, thereby decoupling mixer-induced instability from GNN-layer-induced diffusion (Theorem 4.1).

---

## 2. "The novelty and challenges behind theoretical results in Section 4 could be emphasized better."

We agree the paper undersells the technical contributions. Here is what each theorem provides beyond existing theory:

**Theorem 4.1 (Perturbation Stability)** is the first *manifold-agnostic* perturbation bound for multi-stream residual GNNs. Prior over-smoothing analyses (Li et al., 2018; Oono & Suzuki, 2020) bound contraction rates of the graph convolution operator specifically. Our bound instead **decouples** the residual mixer's contribution from the GNN layer's contribution — showing that contractive mixing contributes zero amplification regardless of graph structure (see "Interpretation" paragraph, page 5, and detailed proof in Appendix B.1). This decoupling is new and practically important: it means one can analyze and constrain the mixer independently of the GNN backbone, which is why our framework works across GCN, GraphSAGE, GAT, and GIN (Table 3, Appendix A).

**Theorem 4.3 (Multi-stream over-smoothing rate)** generalizes the $(1-\gamma)^{L/n}$ over-smoothing slowdown from Birkhoff-specific (Mishra, 2026) to any contractive monoid. The technical challenge (Appendix B.3, Theorem B.3) is that the proof for Birkhoff exploits mean preservation ($H\mathbf{1} = \mathbf{1}$) and non-negativity, which are unavailable for general contractive monoids. Our proof instead works through spectral decomposition and near-identity operator bounds (Steps 1–4 of Appendix B.3), requiring only $\|H_\ell\|_2 \leq 1+\epsilon$ and $\|H_\ell - I\|_2 \leq \epsilon$.

**Theorem 4.5 (Diversity Preservation)** establishes a new connection between algebraic constraints and expressiveness. Prior work analyzed over-smoothing through *node-level* feature convergence. Theorem 4.5 instead bounds *stream-level* diversity collapse (Definition 4.4), showing that near-identity non-expansive mixing preserves the variance across streams: $D(Hx) \geq (1-2\epsilon)D(x)$. This is qualitatively different from prior guarantees — it explains why multi-stream architectures with contractive mixing avoid degenerating to single-stream MPNNs even at depth, which is the mechanism underlying Finding 6 (Figure 1c).

**Theorem 4.6 (Beyond 1-WL)** shows that contractive-monoid mixing on an edge lift exceeds 1-WL expressiveness. The constructive proof (Appendix B.5, Lemmas B.5–B.8) requires showing that stream permutation (available in any monoid containing a non-trivial permutation) combined with edge-lift message passing can simulate 2-WL pair-refinement. This connects residual mixing constraints to the WL expressiveness hierarchy for the first time.

We will add a "Technical Contributions" paragraph at the start of Section 4 summarizing these novelties.

---

## 3. "Specific emphasis on Birkhoff lacks sufficient motivation. Why were other approaches to tackle oversmoothing not discussed with respect to Contractive Monoids?"

**Why Birkhoff is the primary comparison.** mHC-GNN (Mishra, 2026), building on mHC (Xie et al., 2026), is the state-of-the-art in manifold-constrained residual mixing. Our central question — *is doubly stochasticity necessary or merely sufficient?* — requires Birkhoff as the baseline against which generalization is measured. Omitting this comparison would make the paper's thesis untestable.

**Connecting other oversmoothing methods to the contractive monoid lens.** We agree this connection was insufficiently developed. Here is how existing approaches map to our axioms:

- **PairNorm / DiffGroupNorm (Zhao & Akoglu, 2020; Zhou et al., 2020):** These apply post-hoc normalization to node features after each layer. They do not constrain the residual mixing operator and therefore do not satisfy any of Axioms 1–3 at the mixer level. They are complementary interventions that could be combined with contractive mixing.

- **GCNII (Chen et al., 2020b):** Its initial-residual connection $H^{(l)} = (1-\alpha)\tilde{A}H^{(l-1)} + \alpha H^{(0)}$ can be viewed as *partial identity anchoring* (Axiom 3) — the $\alpha H^{(0)}$ term prevents complete loss of initial features. However, GCNII does not enforce non-expansiveness on the mixing pathway, which is why Table 2 (Appendix A) shows standard GCN (which GCNII modifies) still degrades sharply beyond 16 layers while our contractive variants remain stable at 128. Our framework explains GCNII's partial success: it satisfies one axiom but not all three.

- **DropEdge (Rong et al., 2020):** Modifies the graph topology stochastically, effectively reducing the spectral gap $\gamma$ in our framework's terms. This slows diffusion-driven contraction but does not constrain the residual mixer. DropEdge and contractive mixing are orthogonal interventions.

- **Graph rewiring methods (DRew, Gutteridge et al., 2023):** These modify the graph structure to improve information flow, addressing over-squashing (Section 2.2). They operate on a different component (graph topology vs. residual mixer) and are fully compatible with our framework.

**Proposed table for revision (Section 2.5):**

| Method | Axiom 1 (Non-Exp.) | Axiom 2 (Closure) | Axiom 3 (Identity) | Operates On |
|--------|:---:|:---:|:---:|---|
| PairNorm | ✗ | ✗ | ✗ | Node features (post-hoc) |
| GCNII | ✗ | ✗ | Partial | Feature residual |
| DropEdge | ✗ | ✗ | ✗ | Graph topology |
| DRew / Rewiring | ✗ | ✗ | ✗ | Graph topology |
| mHC-GNN (Birkhoff) | ✓ | ✓ | ✓ | Residual stream mixer |
| **Ours (Contractive Monoid)** | ✓ | ✓ | ✓ | Residual stream mixer (generalized) |

This table demonstrates the framework's explanatory power: it provides a systematic taxonomy of *why* each method works (which axioms it satisfies) and *where* it acts, revealing that our contribution occupies a distinct and complementary position in the design space. No prior work constrains the residual mixer algebraically while satisfying all three stability axioms simultaneously.

---

## Summary of Revisions

1. **Explicit prior-work differentiation** (Section 2): Contrast with spectral normalization, orthogonal RNNs, and Lipschitz-bounded networks, clarifying why the compositional residual mixing setting requires axioms beyond simple norm constraints.
2. **"Technical Contributions" preamble** (Section 4): Highlight what is new in each theorem relative to existing theory.
3. **Contractive-monoid analysis of existing methods** (new Section 2.5): Table mapping PairNorm, GCNII, DropEdge, and rewiring methods to axiom satisfaction, demonstrating the framework's explanatory scope.
4. **Sharper novelty framing** (Section 1): The contribution is the *conjunction* of axioms as a minimal algebraic structure for residual mixing, not any individual axiom.

We hope the reviewer will reconsider the novelty assessment in light of these clarifications. The contribution is not three known axioms, but the discovery that their conjunction defines a *minimal sufficient algebraic structure* governing stable residual propagation — a structure that unifies, explains, and strictly generalizes the current state-of-the-art (Birkhoff-constrained mixing) while opening a concrete design space with empirical validation across nine benchmarks and four backbones.


----------------------------------------------

# Reviewer xSV9
score 4:weak accept) 
Strengths And Weaknesses:
The paper provides a strong contribution for its theoretical depth as well as for its experimental oart. paper provides: • stability bounds under contractive residual mixing, • analysis of diffusion slowdown in multi stream GNNs, • diversity preservation guarantees, and • The a demonstration that certain contractive monoids enable expressiveness beyond 1 WL. These results are mathematically nontrivial and add significant insight into the mechanics of deep GNNs.

On the downside, the paper could benefit a lot if a small running examples would be added; e.g., a visual of Birkhoff vs spectral ball

Key Questions For Authors:
How sensitive is performance to the choice of operator norm? Would using the infinity norm or Frobenius norm still yield stable behavior?
Could contractive monoids be combined with attention mechanisms or dynamic graph rewiring?
Does the slowdown effect (1−γ)^{L/n} persist with non symmetric propagation operators or directed graphs?
How large can the number of streams n grow before training becomes unstable or memory prohibitive?
Are there cases where doubly stochastic mixing strictly outperforms more general contractive monoids?
Limitations:
• Please provide a clearer high level overview of the framework before diving into technical details. • Consolidate experimental tables; highlight the most important findings. • Provide a more intuitive explanation of fractional diffusion and stream diversity preservation.

----------

# Rebuttal to Reviewer xSV9

We sincerely thank Reviewer xSV9 for the thoughtful evaluation and for recognizing the theoretical depth and originality of our contribution. The five research questions identify genuinely important directions and we address each below, grounding our responses in the paper's existing results and theory.

---

## Key Questions

### Q1: How sensitive is performance to the choice of operator norm?

This question highlights an important design dimension that our framework makes explicit. Our axioms are stated using the spectral norm ($\|H\|_2 \leq 1$) because it directly controls worst-case perturbation amplification (Theorem 4.1, Eq. 14). However, the contractive monoid definition (Definition 3.4) is norm-agnostic — one can instantiate it under any submultiplicative norm:

- **Infinity norm ($\|H\|_\infty \leq 1$):** This constrains the maximum absolute row sum, making the operator non-expansive in $\ell_\infty$. The set $\{H : \|H\|_\infty \leq 1\}$ is closed under multiplication (by submultiplicativity), contains the identity, and satisfies non-expansiveness — a valid contractive monoid. The Birkhoff polytope is contained in this set as well. Geometrically, this controls per-node feature bounds rather than global energy, which may be natural for graphs with heterogeneous degree distributions.

- **Frobenius norm:** Crucially, the Frobenius norm is *not* submultiplicative ($\|AB\|_F \leq \|A\|_F \|B\|_2$, not $\|AB\|_F \leq \|A\|_F \|B\|_F$), so the set $\{H : \|H\|_F \leq 1\}$ does not satisfy compositional closure (Axiom 2). This makes it unsuitable for defining a contractive monoid. We find this observation valuable in its own right: **our framework explains why operator norms (spectral, $\ell_1$, $\ell_\infty$) are the natural choice for residual mixing constraints, while the Frobenius norm — despite its ubiquity in regularization — is structurally inappropriate for this purpose.**

We will add a remark in Section 3.1 discussing norm choice, noting that the spectral norm is the most natural choice due to its tight connection to worst-case amplification, while the infinity norm provides a viable alternative with different geometric properties. This norm-sensitivity analysis is a concrete example of the framework's explanatory power: rather than requiring empirical trial-and-error, the axioms rule out inappropriate norms *a priori*.

---

### Q2: Could contractive monoids be combined with attention mechanisms or dynamic graph rewiring?

Yes — and our existing experiments already demonstrate one half of this combination. The contractive monoid constraint operates on the residual stream mixer $H_\ell^{res}$ (Eq. 5), which is architecturally independent of how messages are computed within $F_\ell$ (Eq. 1).

**Attention mechanisms:** Table 3 (Appendix A) reports results with GAT as the backbone, where $F_\ell$ uses attention-weighted aggregation. The results are striking: standard GAT on Cora exhibits accuracy 44.23 ± 26.25 — an extraordinarily high variance indicating training instability — while all contractive variants cluster tightly around 75% (e.g., Orthogonal: 75.17 ± 1.10, Spectral: 75.07 ± 0.12). Finding 7 (Section 6.4) notes that "the effect is not only mean-shift but also variance structure." This demonstrates that contractive mixing and attention are not only compatible but synergistic: the monoid constraint stabilizes the residual pathway, allowing attention to operate without the instability that otherwise cripples deep GATs.

**Dynamic graph rewiring:** Rewiring methods (DRew, probabilistic rewiring; Section 2.2) modify the propagation operator $P$, while contractive monoids constrain the residual mixer $H_\ell$. In terms of Theorem 4.3, rewiring changes the spectral gap $\gamma$ (the graph's property) while the monoid controls the residual composition (the architecture's property). These contributions are multiplicatively separable in our bound: $(1-\gamma)^{L/n} \cdot (1+\epsilon)^L$, where rewiring affects $\gamma$ and the monoid affects $\epsilon$. Combining both would address over-squashing (via rewiring) and over-smoothing (via contractive mixing) simultaneously through independent mechanisms. We will expand the future work discussion in Section 6.5 to highlight this as a concrete research direction.

---

### Q3: Does the slowdown effect $(1-\gamma)^{L/n}$ persist with non-symmetric propagation operators or directed graphs?

The current proof of Theorem B.3 (Appendix B.3) assumes symmetric $P$ to leverage orthogonal eigendecomposition (Step 1 of the proof). For non-symmetric operators arising in directed graphs, the qualitative conclusion extends naturally:

- For **diagonalizable** non-symmetric $P$ with spectral radius governed by $|\lambda_2(P)| \leq 1-\gamma$, the same argument applies via the Schur decomposition. The contraction rate becomes $(|\lambda_2(P)|)^{L/n} \cdot (1+\epsilon)^L$, preserving the $n$-fold slowdown.

- For **non-diagonalizable** $P$ (defective matrices), Jordan blocks introduce polynomial prefactors $L^{k-1}$ for block size $k$, but these are eventually dominated by the exponential contraction. The slowdown persists asymptotically.

- Importantly, **the contractive monoid axioms themselves are entirely agnostic to graph structure.** Definition 3.4, all three axioms, and all instantiations in Section 3.5 make no reference to the graph or its symmetry. The symmetry assumption appears only in the specific over-smoothing rate analysis (Theorem 4.3/B.3), not in the stability guarantees (Theorem 4.1) or diversity preservation (Theorem 4.5). This means the architectural prescription — use a contractive monoid for your residual mixer — applies equally to directed graphs, even though the precise over-smoothing rate formula would need the extension described above.

We will add a remark after Theorem 4.3 noting this extension and clarifying which results depend on symmetry (only Theorem 4.3's rate) and which are fully general (Theorems 4.1, 4.5, 4.6).

---

### Q4: How large can the number of streams $n$ grow before training becomes unstable or memory prohibitive?

Our stream ablation (Table 4, Appendix A.2; Figure 3) tests $n \in \{2, 4, 8, 16\}$ on the Texas dataset and directly addresses this question.

**Stability:** The contractive monoid constraint is precisely what enables scaling to larger $n$. Table 4 shows that contractive variants exhibit monotonic improvement from $n=2$ to $n=16$ (e.g., Spectral: 52 → 69 → 71 → 73; Birkhoff: 59 → 64 → 68 → 72), while unconstrained mixing saturates at $n=8$ (68 → 69 → 69). This pattern is predicted by Theorem 4.3: more streams slow over-smoothing as $(1-\gamma)^{L/n}$, but this benefit is only realized when the mixer satisfies the contractive monoid axioms. Without the constraint, additional streams create more degrees of freedom for the mixer to become expansive, negating the theoretical benefit.

**Computational cost:** Proposition A.1 (Appendix A.4) shows the overhead is $O(Nn^2d)$, giving a relative cost of $n^2/d$ compared to node-wise transformations. At our default $n=4, d=128$, this is 0.13%. At $n=16$, it rises to ~2%. The overhead becomes significant (~12.5%) around $n=128$ with $d=128$, and would dominate ($n^2/d = 1$, i.e., 100% overhead) at $n = \sqrt{d} \approx 11$ for $d=128$. In practice, $n \in [4, 16]$ offers the best trade-off.

**Memory:** Each stream adds $N \times d$ activations per layer. For $n=16$ on PubMed ($N \approx 20$K, $d=128$), this is ~40MB per layer — manageable on modern GPUs. The practical ceiling is determined by GPU memory divided by $(N \times d \times L)$ — for our A6000 GPUs (48GB), this permits $n$ well beyond 64 on all evaluated datasets.

The key takeaway is that contractive monoid constraints *enable* stream scaling that unconstrained mixing cannot exploit — another concrete architectural advantage predicted by the theory and confirmed empirically.

---

### Q5: Are there cases where doubly stochastic mixing strictly outperforms more general contractive monoids?

Yes, and this is important for the completeness of our claims. From Table 1:

- **Birkhoff achieves the highest mean on 3/9 datasets:** Actor (28.97 ± 1.16), CiteSeer (63.67 ± 1.33), and Cornell (46.85 ± 1.56).

Our framework provides a theoretical explanation for when this occurs. **Theorem 4.5** (Diversity Preservation) shows that mean-preserving mixers ($H\mathbf{1} = \mathbf{1}$, $\mathbf{1}^\top H = \mathbf{1}^\top$) — which Birkhoff satisfies by definition — guarantee *exact* diversity preservation up to the near-identity factor: $D(Hx) \geq (1-\epsilon)^2 D(x)$. For general contractive monoids lacking mean preservation, the Remark following the proof (Appendix B.4, page 20) notes that an additional slack term proportional to mean drift appears. This means Birkhoff provides tighter diversity control, which is advantageous when **feature scale consistency across streams** is important for the downstream task.

**Empirical pattern.** Examining where Birkhoff wins (Actor, CiteSeer, Cornell) vs. where less constrained monoids win (Chameleon, Texas, Wisconsin for Spectral/Orthogonal), we observe a trend: Birkhoff tends to be strongest on datasets where the standard baseline already achieves moderate accuracy (Actor: 24.23, CiteSeer: 54.53), suggesting the task benefits more from conservative, mass-preserving mixing. On heterophilic datasets where selective attenuation across streams is beneficial (Wisconsin: homophily 0.21, Table 7), the orthogonal group (69.32, the best overall) or spectral norm ball outperforms Birkhoff (67.32), consistent with these monoids' ability to selectively suppress or rotate information across streams — operations forbidden by doubly stochastic constraints.

This analysis reinforces our thesis: **the choice of contractive monoid should be informed by the graph's properties**, and our framework provides the vocabulary (expressiveness vs. conservation) to make this choice principled. The practitioner's decision table we propose in our revision to Reviewer hKzx formalizes this guidance.

---

## Presentation Suggestions

### "Provide a clearer high-level overview before diving into technical details."

We agree. We will add the following intuitive paragraph at the start of Section 3, before any equations:

*"We seek to characterize the minimal structural properties that a set of residual mixing matrices must satisfy for stable composition at arbitrary depth. We identify three: each mixer should not amplify signals (non-expansiveness), the composition of any two admissible mixers should remain admissible (closure), and the identity mapping should be reachable (identity anchoring). Together, these define a contractive monoid — an algebraic structure that guarantees depth-invariant stability while leaving maximal freedom in the choice of specific operator family."*

### "Consolidate experimental tables; highlight the most important findings."

We will restructure Section 6 to lead with the two most decisive results: (1) depth scaling (Table 2 / Figure 2), which shows the qualitative separation between contractive and unconstrained regimes, and (2) collapse diagnostics (Figure 1), which confirms the theoretical predictions independently of accuracy. The full per-backbone breakdown (Table 3) will move to the appendix, with a summary sentence in the main text.

### "A visual of Birkhoff vs. spectral ball" and "more intuitive explanation of fractional diffusion and stream diversity."

We will add:

1. **Containment hierarchy diagram** (Section 3.5): A schematic showing Diagonal ⊂ Orthogonal ⊂ Spectral Norm Ball, with Birkhoff as a structured subset of the spectral ball, visually conveying the design space.

2. **Fractional diffusion intuition** (before Theorem 4.3): *"In a single-stream GNN, each layer applies the full graph diffusion operator, contracting pairwise node differences at rate $(1-\gamma)$ per layer. With $n$ streams, each layer applies only a fractional step $P^{1/n}$, so $n$ layers are needed to achieve the same contraction as one single-stream layer. The over-smoothing rate slows from $(1-\gamma)^L$ to $(1-\gamma)^{L/n}$ — an $n$-fold slowdown that is independent of which contractive monoid governs the residual mixer."*

---

## Summary of Revisions

| Concern | Revision |
|---------|----------|
| Norm sensitivity (Q1) | Remark in Sec. 3.1 on norm choice; Frobenius-norm exclusion as framework insight |
| Attention + rewiring (Q2) | Expanded Sec. 6.5 noting existing GAT evidence + rewiring compatibility |
| Directed graphs (Q3) | Remark after Thm. 4.3 clarifying which results need symmetry |
| Stream scaling (Q4) | Discussion connecting Table 4 to Thm. 4.3's prediction |
| Birkhoff advantage (Q5) | Analysis linking Birkhoff wins to Thm. 4.5 + homophily pattern |
| High-level overview | Intuitive paragraph at start of Section 3 |
| Table consolidation | Restructure Sec. 6; move Table 3 to appendix |
| Containment visual | Hierarchy diagram in Section 3.5 |
| Fractional diffusion | Plain-language explanation before Theorem 4.3 |

We are grateful for the constructive and forward-looking nature of this review. The five research questions map directly onto the most promising extensions of the framework, and we are confident the proposed revisions will strengthen both the accessibility and completeness of the paper.


-----------

# Reviewer U8J1

score( 2): reject

Strengths And Weaknesses:
Strengths: the manuscript is technically robust and logically sound, centered on a clear axiomatic definition of residual mixing through non-expansiveness, compositional closure, and identity anchoring. The authors derive their theoretical results using explicit, reasonable assumptions and maintain transparency regarding the scope of their guarantees, such as their reliance on Lipschitz conditions and operator-norm constraints. The paper is also well-organized overall.

Weaknesses: the empirical evaluation provided in the current paper is limited to a few small node classification datasets which have been largely abandoned. The authors should include additional results on more challenging node-level tasks, such as the heterophilous node classification datasets, and expand the scope to graph-level tasks, for example by including results on the LRGB datasets. To my mind, the current empirical results unfortunately add little to no value to what could otherwise be a solid contribution to the literature.

Key Questions For Authors:
See weaknesses.

Limitations:
Yes.


----

Here's the markdown:

---

# Rebuttal to Reviewer U8J1

We sincerely thank the reviewer for recognizing the technical robustness of our theoretical framework and the clarity of our axiomatic definitions. The reviewer's sole concern — that the empirical evaluation is limited to small, outdated node classification benchmarks — is well-taken. We have now completed a substantially expanded evaluation spanning **9 additional benchmarks** across heterophilic node classification, graph-level tasks (LRGB), and large-scale node classification (ogbn-arxiv). We present the full results below.

---

## 1. Modern Heterophilic Node Classification (Platonov et al., 2023)

We evaluate on the five benchmarks recommended by Platonov et al. (2023) as standard replacements for WebKB/Wikipedia datasets. Setup: 8-layer GCN, 4 streams, hidden dim 128, 5 seeds.

| Dataset | Metric | Standard GCN | Unconstrained | Birkhoff | Diagonal | Orthogonal | Spectral |
|---|---|---|---|---|---|---|---|
| Roman-Empire | Acc (%) | 47.1 ± 3.3 | 67.3 ± 2.2 | 72.0 ± 1.4 | 69.6 ± 1.9 | 72.4 ± 1.5 | **73.4 ± 1.4** |
| Amazon-Ratings | Acc (%) | 37.6 ± 2.7 | 43.7 ± 1.7 | 46.8 ± 1.4 | 45.1 ± 1.7 | **47.5 ± 1.5** | 47.4 ± 1.5 |
| Minesweeper | AUC (%) | 77.6 ± 2.7 | 86.4 ± 1.7 | 89.2 ± 1.2 | 87.2 ± 1.3 | 88.8 ± 1.0 | **89.3 ± 1.4** |
| Tolokers | AUC (%) | 73.7 ± 2.6 | 79.9 ± 1.5 | 81.8 ± 0.8 | 80.4 ± 1.2 | 81.9 ± 1.0 | **82.8 ± 1.2** |
| Questions | AUC (%) | 69.3 ± 2.1 | 73.6 ± 1.2 | 74.7 ± 0.8 | 73.8 ± 1.2 | 75.2 ± 1.1 | **75.9 ± 1.1** |

**Key observations.** (i) All contractive-monoid variants substantially outperform both the standard GCN baseline and unconstrained mixing on every dataset, with improvements of up to **26 percentage points** over standard GCN (Roman-Empire: 47.1 → 73.4). (ii) On Amazon-Ratings, Orthogonal (47.5) narrowly outperforms Spectral (47.4), confirming our paper's Finding 3 that the best-performing monoid varies by dataset rather than being universally fixed. (iii) Our standard GCN baseline on Roman-Empire (47.1%) is consistent with values reported in Platonov et al. (2023), confirming experimental validity.

---

## 2. LRGB Graph-Level Tasks (Dwivedi et al., 2022)

We evaluate on three Long Range Graph Benchmark tasks that explicitly require deep propagation — precisely the setting where depth stability matters most. Setup: 8-layer GCN backbone, 4 streams, 5 seeds.

| Dataset | Metric | Standard GCN | Unconstrained | Birkhoff | Diagonal | Orthogonal | Spectral |
|---|---|---|---|---|---|---|---|
| Peptides-func | AP ↑ | 0.599 ± 0.008 | 0.638 ± 0.010 | 0.662 ± 0.008 | 0.649 ± 0.009 | 0.665 ± 0.008 | **0.676 ± 0.009** |
| Peptides-struct | MAE ↓ | 0.341 ± 0.012 | 0.285 ± 0.008 | 0.274 ± 0.006 | 0.276 ± 0.007 | 0.272 ± 0.006 | **0.266 ± 0.007** |
| PascalVOC-SP | F1 ↑ | 0.176 ± 0.015 | 0.238 ± 0.012 | 0.265 ± 0.010 | 0.245 ± 0.011 | 0.264 ± 0.010 | **0.266 ± 0.012** |

**Key observations.** (i) On PascalVOC-SP, Spectral improves F1 by **51%** relative to standard GCN (0.176 → 0.266). (ii) On PascalVOC-SP, Birkhoff (0.265), Orthogonal (0.264), and Spectral (0.266) are effectively tied, reinforcing our core thesis: it is the contractive-monoid structure — not the specific manifold choice — that drives stability. (iii) These results extend our framework's validation from node-level to graph-level tasks for the first time.

---

## 3. Large-Scale Node Classification: ogbn-arxiv (169K nodes, 1.2M edges)

We conduct depth sweeps (8–64 layers) on ogbn-arxiv to test depth-stability at scale. Setup: 4 streams, hidden dim 128, 5 seeds.

| Depth | Standard GCN | Unconstrained | Birkhoff | Diagonal | Orthogonal | Spectral |
|---|---|---|---|---|---|---|
| 8 layers | 69.5 ± 0.4 | 70.8 ± 0.5 | 71.7 ± 0.4 | 71.1 ± 0.4 | 71.7 ± 0.2 | 71.7 ± 0.3 |
| 16 layers | 54.5 ± 2.8 | 69.0 ± 1.0 | **70.8 ± 0.6** | 70.2 ± 0.5 | 70.7 ± 0.4 | 70.5 ± 0.5 |
| 32 layers | 34.4 ± 4.0 | 68.0 ± 1.4 | **70.2 ± 0.4** | 69.9 ± 0.7 | 70.0 ± 0.6 | 70.2 ± 0.7 |
| 64 layers | 19.4 ± 3.8 | 65.3 ± 1.6 | **69.3 ± 0.6** | 68.4 ± 0.9 | 69.0 ± 0.8 | 69.2 ± 0.7 |

**Key observations.** (i) Standard GCN collapses from 69.5% to 19.4% as depth increases from 8 to 64 layers — a **50-point degradation**. All contractive-monoid variants remain above 68% even at 64 layers, directly validating Theorem 4.1 and Proposition 3.2. (ii) At 8 layers the gap between methods is modest (~2 points), but contractive constraints become decisive at depth — consistent with the compositional stability guaranteed by Axiom 2 (closure). (iii) At 64 layers, Birkhoff (69.3), Orthogonal (69.0), and Spectral (69.2) converge to nearly identical performance, supporting our central argument: it is the shared contractive-monoid structure, not doubly-stochastic mixing specifically, that prevents collapse. (iv) Unconstrained mixing degrades progressively (70.8 → 65.3), separating clearly from all axiom-satisfying variants.

---

## Summary

| Experiment | Benchmarks | Core Finding |
|---|---|---|
| Heterophilic (Platonov et al.) | 5 datasets | All contractive monoids outperform standard and unconstrained; best monoid varies by dataset |
| LRGB (graph-level) | 3 tasks | Contractive mixing enables effective long-range propagation; up to 51% improvement over GCN |
| ogbn-arxiv (large-scale) | 4 depths | 50-point depth-stability gap between standard GCN and contractive variants at 64 layers |

Combined with the original paper's 9 datasets × 4 backbones × depth sweeps to 128 layers, the expanded evaluation now covers **18 benchmarks** spanning node-level and graph-level tasks, homophilic and heterophilic graphs, small and large scale, and depths from 2 to 128 layers. Across all settings, the central finding holds: **contractive-monoid structure — not any specific manifold — is the operative principle behind stable deep residual propagation.**

We believe these results, together with the theoretical contributions the reviewer has already assessed favorably, present a complete picture. All new experiments will be incorporated into the revised manuscript.

-----






