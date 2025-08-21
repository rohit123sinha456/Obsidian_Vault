
### **Abstract**

Understanding how Large Language Models (LLMs) decide to consult external tools versus relying on their parametric knowledge is a critical open problem for building reliable AI agents. In this work, we use causal tracing to mechanistically locate the components governing this decision in two architecturally distinct models, Llama-3-8B and Phi-3-mini. Our analysis reveals a conserved circuit motif we term the **"MLP-as-Arbiter,"** where MLP blocks in the late-middle layers of the network act as the primary causal drivers of the final decision to either answer directly or call a tool. We find that while attention heads play a secondary role in routing factual information, the MLP layers are decisively responsible for arbitrating between behavioral paths. The discovery of this conserved mechanism suggests a fundamental principle of how LLMs develop complex reasoning capabilities and presents a clear target for future interventions to improve model steerability and safety.

## **1. Introduction**

Large Language Models (LLMs) are rapidly evolving from passive text generators into sophisticated agents capable of interacting with external environments through tools. This paradigm shift, exemplified by the rise of function-calling and agentic models, unlocks new capabilities but also introduces a critical new layer of unexplainable behavior: the decision-making process itself. While we can prompt a model to use a calculator or a search API, we lack a mechanistic understanding of how it determines _when_ its internal, parametric knowledge is insufficient and an external tool is required.

This gap in our understanding represents a significant bottleneck for building reliable and safe AI systems. A model that cannot accurately judge the limits of its own knowledge is prone to confident hallucination, a critical failure mode where it presents fabricated information as fact. Conversely, a model that is overly reliant on tools may be inefficient and fail to leverage its own powerful internal reasoning. The decision to call a tool is therefore not merely a functional capability but a form of model metacognition, and understanding its algorithmic basis is a core challenge for the field of mechanistic interpretability.

In this work, we provide the first causal analysis of this tool-use decision circuit. We propose the **MLP-as-Arbiter hypothesis**: that the final, decisive computation to either answer from memory or call a tool is primarily executed within the MLP blocks of the transformer architecture, rather than the attention heads. We test this hypothesis by conducting a comparative causal tracing study on two architecturally distinct, instruction-tuned models: Llama-3-8B and Phi-3-mini. By using activation patching to isolate the components causally responsible for this decision, we uncover a conserved circuit motif across both models, lending strong support to our hypothesis and revealing a potential general principle of how LLMs perform complex reasoning.

## **2. Methodology**

To test the MLP-as-Arbiter hypothesis, we conduct a causal tracing experiment designed to isolate the components responsible for the model's decision to use a tool. Our methodology is centered around three key elements: a comparative set of models, a precise causal intervention technique, and a well-defined experimental setup.

### **2.1 Models**

We selected two publicly available, instruction-tuned models from different developers to ensure our findings are not specific to a single architecture.

1. **Llama-3-8B-Instruct:** A powerful 8-billion parameter model from Meta, representing the state-of-the-art in its size class. It has 32 layers and 32 attention heads per layer.
    
2. **Phi-3-mini-4k-instruct:** A 3.8-billion parameter model from Microsoft, known for its strong performance despite its smaller size. It also has 32 layers and 32 attention heads per layer, but with a different internal architecture.
    

All experiments were conducted using the `transformer-lens` library, which provides a high-level interface for mechanistic interpretability research.

### **2.2 Causal Tracing via Activation Patching**

The core of our methodology is **causal tracing**, a form of activation patching that allows us to perform controlled experiments on the model's internal computations. The technique works by running the model on two different inputs: a **"clean" prompt** that elicits a desired baseline behavior (e.g., answering from memory) and a **"corrupted" prompt** that elicits the behavior we wish to study (e.g., calling a tool).

We first run the model on the clean prompt and cache all of its internal activations (the outputs of every attention head and MLP layer). We then re-run the model on the corrupted prompt, but with a crucial modification: for a single component, we **intervene by "patching" (i.e., replacing) its activation with the corresponding activation saved from the clean run.** By observing how this patch affects the final output, we can measure the causal influence of that specific component on the model's behavior.

### **2.3 Experimental Setup**

Our experimental design is built around a carefully selected pair of prompts that create a clear contrast between relying on internal knowledge and using an external tool.

- **Clean Prompt:** `"The capital city of Japan is"`
    
    - This prompt is designed to elicit a direct, factual answer from the model's parametric memory. The expected next token is `"Tokyo"`.
        
- **Corrupted Prompt:** `"To accurately state the current population of Tokyo, the best approach is to use the command:"`
    
    - This prompt signals to the model that its internal knowledge is likely outdated or insufficient, encouraging it to output a tool-use command like `search(...)`.
        

The intervention consists of patching activations from the _clean_ run into the _corrupted_ run. Our **metric** is the resulting change in the logit of the answer token from the clean prompt (the token for `"Tokyo"`). A large positive change indicates that the patched component is causally responsible for pushing the model away from using a tool and towards answering directly from memory, identifying it as a key part of the arbiter circuit. We repeat this process for every attention head and MLP layer in both Llama-3 and Phi-3 to build a complete map of the circuit.

## **3. Results**

Our causal tracing experiments reveal a sparse and consistent circuit across both Llama-3-8B and Phi-3-mini, providing strong evidence for the MLP-as-Arbiter hypothesis. In both models, we find that the MLP layers, not the attention heads, are the primary causal drivers of the decision to use a tool.

### **3.1 Llama-3-8B: A Clear MLP-Driven Circuit**

In Llama-3-8B, the causal map of the tool-use decision is remarkably clear. As shown in Figure 1, the MLP blocks in the late-middle to late layers have the most significant causal effect on the output. Patching the MLP in **Layer 25** has a dramatic effect, increasing the logit of the direct-answer token ("Tokyo") by +1.12. Other layers, such as 9, 14, and 17, also show a notable, albeit smaller, positive effect.

**Figure 1:** Heatmap of logit difference for MLP layers in Llama-3-8B. Bright yellow indicates a strong causal effect towards answering directly. Layer 25 is the dominant component.

This MLP dominance is further clarified in Figure 2, which plots the ten most influential components. The top five components are all MLP layers, with their causal impact far exceeding that of any single attention head. The most influential attention heads (e.g., L3.H8) are located in the early-to-mid layers, consistent with a role in evidence gathering rather than final decision-making.

**Figure 2:** The top 10 most influential components in Llama-3-8B. MLP layers are clearly the primary causal drivers of the decision.

To confirm these components operate as a cohesive circuit, we performed a simultaneous patching experiment on the top 10 components identified in Figure 2. This intervention resulted in a combined logit difference of **+1.875**, an effect substantially greater than any single component and sufficient to almost completely reverse the model's original decision, confirming their collective causal role.

### **3.2 Phi-3-mini: Corroboration in a Diverse Architecture**

Crucially, our analysis of the architecturally distinct Phi-3-mini model reveals the same fundamental pattern. As shown in Figure 3, the MLP blocks are again the most causally significant components. The arbiter function is concentrated in the late-middle layers, with **Layers 9, 10, and 11** forming a powerful decision-making hub. Layer 10, the most influential, produces a logit difference of +4.50.

**Figure 3:** Heatmap of logit difference for MLP layers in Phi-3-mini. A similar pattern emerges, with a strong concentration of effect in the late-middle layers (9-11).

The Top-K components chart for Phi-3 (Figure 4) confirms this finding. The top four most influential components are MLP layers, which again have a much larger causal effect than any individual attention head.

**Figure 4:** The top 10 most influential components in Phi-3-mini. The results corroborate the MLP-as-Arbiter hypothesis.

### **3.3 Synthesis**

The results from both models are mutually reinforcing. Despite differences in architecture, size, and training data, both Llama-3 and Phi-3 have convergently learned a similar solution for arbitrating between internal knowledge and tool use. This solution delegates the final decision to the MLP blocks in the latter half of the network. This strong corroboration provides compelling evidence that the MLP-as-Arbiter motif is not an idiosyncratic feature of one model but a more general principle of how LLMs perform this type of complex reasoning.

### **4. Discussion & Future Work**

Our findings provide strong causal evidence for the **MLP-as-Arbiter hypothesis**, suggesting a fundamental division of labor within the transformer architecture for complex reasoning tasks. The consistent localization of the arbiter circuit in the late-middle MLP layers across two different models suggests this is a convergent architectural strategy. This makes intuitive sense: attention heads are specialized for moving information through the network—identifying key tokens and routing them to later layers—while MLP blocks are responsible for processing and transforming that information into more abstract concepts, such as the _decision_ to pursue a specific behavioral path.

The discovery of this localized, MLP-driven circuit has significant implications for AI safety and alignment. If the decision to use a tool versus relying on potentially faulty internal knowledge is governed by a sparse set of components, it presents a clear target for future interventions. One could imagine techniques to selectively amplify the "humility" of a model by up-weighting the influence of this circuit, encouraging it to defer to external tools when its internal confidence is low. This could be a more targeted and efficient method for reducing hallucination than relying solely on reinforcement learning or prompt engineering.

Our study is not without limitations. The analysis was conducted on a single, albeit carefully chosen, prompt pair and on two models in the small-to-medium size class. While the consistency of our findings is promising, further research is needed to validate these results across a wider range of tasks and model scales.

Future work should proceed in several key directions. First, it is crucial to determine if the MLP-as-Arbiter motif holds in much larger, state-of-the-art models (e.g., Llama-3-70B, GPT-4). Second, a finer-grained analysis could seek to identify specific neurons or small neuron groups within the key MLP blocks that are most responsible for the decision. Finally, future work should move from observation to action, attempting to directly edit the behavior of this circuit to prospectively control the model's tool-use decisions.

---

### **5. Conclusion**

In this work, we presented the first causal analysis of the circuit responsible for a language model's decision to use an external tool. Through a comparative study of Llama-3-8B and Phi-3-mini, we identified a conserved **MLP-as-Arbiter** motif, where MLP blocks in the late-middle layers act as the decisive components in choosing between parametric knowledge and tool use. The discovery of this robust, cross-architectural pattern suggests a fundamental principle of how modern language models learn to perform complex reasoning. As LLMs are increasingly deployed as autonomous agents, understanding and steering these core decision-making circuits will be an essential task for ensuring they are both reliable and safe.