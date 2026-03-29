#### Reviewer hKzx

**Summary:**

This paper proposes a unifying framework for stable residual propagation in Graph Neural Networks (GNNs). Recent work has explored the design of residual matrices using doubly stochastic matrices. The main claim of this paper is that a more general framework, termed contractive monoids, exists, which significantly extends beyond doubly stochastic matrices and can be used to design residual matrices. Several theoretical results are obtained to provide guarantees for stability (Theorem 4.1), over-smoothing mitigation (Theorem 4.3), and expressiveness (Theorem 4.5) for this class of residual matrices. In particular, the paper focuses on three classes of contractive monoids: (i) orthogonal matrices, (ii) diagonal contractive matrices, and (iii) matrices within the spectral norm ball. Extensive experiments on small-, medium-, and large-scale GNN benchmarks demonstrate the effectiveness of these contractive monoids in designing residual matrices that achieve high test accuracy while avoiding representation collapse as the network depth increases.

**Strengths And Weaknesses:**

**Strengths**

- _Originality_: The core idea of this paper is novel. In particular, it generalizes existing approaches for designing residual connections in GNNs beyond the use of doubly stochastic matrices. The introduction of _contractive monoids_ provides a mathematically rigorous framework for this generalization (although I have some concerns regarding the presentation of this idea, which I discuss in the weaknesses section). Extensive experimental results demonstrate the empirical effectiveness of this approach: leveraging contractive monoids leads to improved accuracy in GNNs with residual connections while mitigating over-smoothing as the network depth increases.
    
- _Experimental analysis_: Another strong aspect of this paper is its extensive experimental evaluation. The authors consider a diverse set of benchmarks spanning small-, medium-, and large-scale graphs, which provides a comprehensive assessment of the proposed framework across different regimes. They include comparisons against strong baselines, allowing for a clear evaluation of performance gains. In addition, the paper goes beyond standard accuracy metrics by systematically analyzing depth scalability, demonstrating how the proposed residual designs alleviate performance degradation in deeper GNNs. The experiments also investigate representation quality and over-smoothing effects, providing empirical evidence that the contractive monoid framework preserves expressive node representations as depth increases. Furthermore, they also study the isolated contribution of different classes of contractive monoids (e.g., orthogonal, diagonal contractive, and spectral norm–bounded matrices), offering insight into their individual roles. Overall, the experimental section is thorough, well-structured, and supports the main claims of the paper.


#### Reviewer E1PX

**Summary:**

This paper discusses framework of Contractive Monoids for stable residual propagation in GNNs. This framework is governed by three axioms: nonexpansiveness, compositional closure, and identity anchoring. Furthermore, the authors claim that the recently proposed mHC-GNNs (where residual mixing matrices are restricted to the Birkhoff polytope of doubly stochastic matrices) conform to the principles of Contractive Monoids. A brief study of three representative instantiations of the residual mixing matrix in GNNs is provided with the perspective of Contractive Monoids.

**Strengths And Weaknesses:**

**Strengths**

1. The paper positions Contractive Monoid as a fundamental principle to tackle the oversmoothing problem in GNNs.
2. Experiments demonstrate the utility of adopting Contractive Monoid in practice, where GNNs affirming to these principles significantly outperform others.
3. Paper clarity is good.
   

#### Reviewer xSV9
**Summary:**

Graph neural network shows accuracy needs deep search, but adding layers leads to oversmoothing. This paper investigates whether doubly stochastic residual mixing (as used in manifold constrained hyper connections, mHC) can stabilizes very deep graph neural networks (GNNs). The authors suggest that stability does not fundamentally arise from doubly stochasticity but rather from a broader algebraic structure they call a contractive monoid—a set of mixing operators satisfying non expansiveness, compositional closure, and identity anchoring.

**Strengths And Weaknesses:**

The paper provides a strong contribution for its theoretical depth as well as for its experimental oart. paper provides: • stability bounds under contractive residual mixing, • analysis of diffusion slowdown in multi stream GNNs, • diversity preservation guarantees, and • The a demonstration that certain contractive monoids enable expressiveness beyond 1 WL. These results are mathematically nontrivial and add significant insight into the mechanics of deep GNNs.

On the downside, the paper could benefit a lot if a small running examples would be added; e.g., a visual of Birkhoff vs spectral ball


#### Reviewer U8J1
**Summary:**

This paper investigates the underlying causes of instability and over-smoothing in deep Graph Neural Networks (GNNs), even when standard residual connections are employed. The authors ask whether the effectiveness of doubly stochastic mixing in recent high-performing models is a unique result of that specific constraint or a byproduct of a broader structural rule. To answer this, their framework focuses on non-expansiveness (preventing signal amplification), compositional closure (ensuring the mixing set remains stable throughout repeated layers), and identity anchoring (incorporating the identity transform to ensure stable initialization). Collectively, these properties form what the authors define as a contractive monoid.

The study demonstrates that this framework is not limited to doubly stochastic matrices but also encompasses diagonal contractions, the spectral norm ball, and orthogonal matrices. Their theoretical results indicate that these specific constraints effectively maintain gradient stability and decelerate over-smoothing.

**Strengths And Weaknesses:**

Strengths: the manuscript is technically robust and logically sound, centered on a clear axiomatic definition of residual mixing through non-expansiveness, compositional closure, and identity anchoring. The authors derive their theoretical results using explicit, reasonable assumptions and maintain transparency regarding the scope of their guarantees, such as their reliance on Lipschitz conditions and operator-norm constraints. The paper is also well-organized overall.