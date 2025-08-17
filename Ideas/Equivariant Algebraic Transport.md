## 30-60-90 day plan

- **Week 1–2**: swap `max`→learned commutative operator; add associativity loss; implement continuous ρM(θ)\rho_M(\theta)ρM​(θ).
    
- **Week 3–4**: plug in dSprites/Multi-dSprites/CLEVR; write evaluation harness (equivariance curve, retrieval@k, IoU).
    
- **Week 5–6**: baselines (Slot Attention, E(2)-CNN backbone, equivariant VAE); run ablations.
    
- **Week 7–8**: polish figures, failure analyses, theory appendix (even if just well-argued identifiability discussion).
    
- **Week 9–10**: draft + workshop submission..


## Paper framing (if you go for it)

**Title**: _Equivariant Algebraic Transport: Learning Group-Consistent Compositional Operators in Latent Space_  
**Contributions**

1. A learned latent group action ρM\rho_MρM​ and mapper ϕ\phiϕ that **transport** image-space symmetry and composition to latent space.
    
2. A **commutator-based metric** for algebra–equivariance compliance.
    
3. A **learned commutative operator** in MMM with associativity regularization that outperforms slot/equivariant baselines on object composition tasks.
    
4. (Optional) Theoretical conditions for identifiability/gauge invariance.  
    **Experiments**
    

- Benchmarks + baselines + ablations as above.
    
- Qualitative demos: latent orbit generation; composition order tests; failure cases.


**Claim**: EAT enables **compositional, equivariant** latent ops that beat object-centric/equivariant baselines on standard datasets.  
**Benchmarks**

- **Object decomposition/composition**: dSprites, Multi-dSprites, Tetrominoes, CLEVR objects (with overlaps and noise).
    
- **Transforms**: C4/C8 rotations, flips, translations; optionally continuous SO(2) via a small Lie algebra layer.  
    **Metrics**
    
- **Equivariance**: average commutator error across group elements and radii of transformations.
    
- **Decomposition**: top-k retrieval accuracy of parts; IoU/Jaccard of reconstructed components; ARI if you add masks.
    
- **Reconstruction quality**: PSNR/SSIM, KID/FID (feature-space).
    
- **Generalization**: out-of-distribution rotations/scales, unseen compositions.  
    **Baselines**
    
- Slot Attention / MONet / IODINE (composition, parts retrieval).
    
- Equivariant VAEs / E(2)-CNN backbones (equivariance).
    
- A naive latent-consistency loss (no learned ρM\rho_MρM​) to show your ρM\rho_MρM​ matters.  
    **Ablations**
    
- Remove algebraic loss; remove equivariance loss; replace max with learned commutative/associative operator (e.g., fM(ma,mb)=σ(W[ma,mb])f_M(m_a,m_b)=\sigma(W[m_a,m_b])fM​(ma​,mb​)=σ(W[ma​,mb​]) trained with associativity regularizer).
    
- Discrete vs. continuous group; tied vs. untied ρM\rho_MρM​ heads; varying mapped dimension.





# identifiability (concise theory sketch)

**Setup.** Let XXX be images; G=SO(2)G=\mathrm{SO}(2)G=SO(2) (or a compact subgroup) acts on XXX by spatial transforms TθT_\thetaTθ​. A (frozen) encoder E:X ⁣→ ⁣LE:X\!\to\!LE:X→L induces an unknown representation ρL(θ)\rho_L(\theta)ρL​(θ) by E(Tθx)≈ρL(θ) E(x)E(T_\theta x)\approx \rho_L(\theta)\,E(x)E(Tθ​x)≈ρL​(θ)E(x). We learn a mapper ϕ:L ⁣→ ⁣M\phi:L\!\to\!Mϕ:L→M, a continuous action ρM:G→O(M)\rho_M:G\to O(M)ρM​:G→O(M), and a binary operator fM:M×M ⁣→ ⁣Mf_M:M\times M\!\to\!MfM​:M×M→M to “transport” the image algebra (here: set-union-like composition) and symmetry:

ϕ∘ρL(θ)  ≈  ρM(θ)∘ϕ,ϕ(L(x∪y))  ≈  fM(ϕ(E(x)), ϕ(E(y))).\phi\circ \rho_L(\theta) \;\approx\; \rho_M(\theta)\circ \phi,\qquad \phi(L(x\cup y)) \;\approx\; f_M(\phi(E(x)),\,\phi(E(y))).ϕ∘ρL​(θ)≈ρM​(θ)∘ϕ,ϕ(L(x∪y))≈fM​(ϕ(E(x)),ϕ(E(y))).

**Assumptions (mild).**

1. GGG is compact and acts continuously on data; training covers a dense set of angles Θ⊂[0,2π)\Theta\subset[0,2\pi)Θ⊂[0,2π).
    
2. ρL\rho_LρL​ decomposes into a sum of real 2-D rotation irreps and (optionally) trivial 1-D blocks (standard for planar rotations).
    
3. ϕ\phiϕ is **injective on the support of E(X)E(X)E(X)** and smooth; ρM(θ)∈O(d)\rho_M(\theta)\in O(d)ρM​(θ)∈O(d) is a continuous group homomorphism.
    
4. fMf_MfM​ is **commutative and associative** on ϕ(E(X))\phi(E(X))ϕ(E(X)), and training enforces these identities on random triples.
    

**Claim (identifiability, up to gauge).**  
If the above holds, then ρM\rho_MρM​ is _unique up to a block-orthogonal change of basis that commutes with the action_ (i.e., within each 2-D rotation plane you can rotate by a constant phase; trivial blocks can permute/scale). Moreover, fMf_MfM​ is identified **on the orbit-closure** of ϕ(E(X))\phi(E(X))ϕ(E(X)) up to the same gauge, because the commutator loss

∥fM(ρM(θ)ma,ρM(θ)mb)−ρM(θ)fM(ma,mb)∥\big\|f_M(\rho_M(\theta)m_a,\rho_M(\theta)m_b)-\rho_M(\theta)f_M(m_a,m_b)\big\|​fM​(ρM​(θ)ma​,ρM​(θ)mb​)−ρM​(θ)fM​(ma​,mb​)​

forces fMf_MfM​ to live in the commutant of ρM\rho_MρM​, and associativity on triples collapses remaining degrees of freedom.

**Gauge-fixing in practice.**  
We (i) parameterize ρM\rho_MρM​ with **explicit 2-D rotation blocks** in a learned orthogonal basis QQQ and penalize ∥Q⊤Q−I∥F2\|Q^\top Q-I\|_F^2∥Q⊤Q−I∥F2​;  
(ii) use **symmetric features** for fMf_MfM​ so commutativity is by construction; and (iii) add associativity and commutator penalties. This pins down ρM\rho_MρM​ (up to per-plane phase) and fMf_MfM​ on the data manifold.

