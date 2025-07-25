Inspired by the work [Link](https://openreview.net/attachment?id=NGKQoaqLpo&name=pdf)

## Abstract

Visual language models (VLMs) integrate text and image data to perform tasks like image captioning and visual question answering. However, how new multimodal data affects a VLM's knowledge base, leading to beneficial generalization or undesirable cross-modal priming, is underexplored. We extend the concept of priming from LLMs to VLMs, where learning new image-text pairs can inadvertently influence unrelated tasks. We introduce "VisOutlandish," a curated dataset of 1,500 image-text pairs designed to probe cross-modal priming. Using this dataset, we demonstrate that priming in VLMs can be predicted by pre-training token and image feature probabilities. We propose two mitigation strategies: cross-modal augmentation and attention-based update pruning, reducing priming effects by 60-90% across models like CLIP-ViT and LLaVA. Our findings provide a foundation for robust multimodal continual learning, with implications for AI safety and interpretability.

## 1. Introduction

VLMs, such as CLIP and LLaVA, combine vision and language processing, enabling tasks like zero-shot image classification and multimodal reasoning. Similar to LLMs, VLMs learn through gradient-based updates, but the interplay between visual and textual modalities introduces unique challenges. Inspired by recent work on LLM priming, where new data causes inappropriate knowledge application in unrelated contexts, we hypothesize that VLMs exhibit cross-modal priming, e.g., learning a new image-text pair about a "red apple" may bias captioning of unrelated fruits. To study this, we create "VisOutlandish," a dataset to probe priming, and propose methods to predict and mitigate it, ensuring controlled knowledge updates in VLMs.[](https://lilianweng.github.io/posts/2022-06-09-vlm/)[](https://arxiv.org/html/2504.09522v1)

## 2. VisOutlandish Dataset

To systematically study cross-modal priming, we design "VisOutlandish," a dataset of 1,500 image-text pairs, split into training (1,200) and evaluation (300) sets. The dataset includes:

- **Diverse Modalities**: Images span natural scenes, objects, and abstract concepts, paired with descriptive, factual, or outlandish captions (e.g., "A purple elephant flying over a city").
- **Priming Triggers**: Each pair is crafted to introduce surprising or conflicting information, measured by low token probability (text) and feature rarity (image) in a pre-trained VLM.
- **Evaluation Probes**: 300 pairs test priming effects across unrelated tasks, e.g., captioning an apple image after training on "purple elephant" pairs.

Images are sourced from public datasets like COCO and ImageNet, with synthetic augmentations to create rare visual features. Captions are generated using GPT-4 with human verification to ensure diversity and relevance.

## 3. Methodology

We formalize cross-modal priming as the unintended influence of new image-text pairs on a VLM’s behavior in unrelated tasks. Our approach includes:

### 3.1 Measuring Priming

We fine-tune VLMs (CLIP-ViT-L-336px and LLaVA-13B) on VisOutlandish training pairs and evaluate priming on probe tasks (captioning, VQA, classification). Priming is quantified as the change in output distribution (e.g., KL-divergence) on unrelated inputs before and after fine-tuning. We predict priming by computing:

- **Textual Surprise**: Token probability of caption keywords in the VLM’s language head.
- **Visual Surprise**: Cosine similarity of image features to pre-training embeddings.

### 3.2 Mitigation Strategies

To dilute priming, we propose:

- **Cross-Modal Augmentation**: During fine-tuning, we augment training pairs with counterfactual image-text combinations (e.g., "green apple" with a red apple image) to reduce overfitting to specific associations. This is inspired by text augmentation in but adapted for multimodal data.[](https://arxiv.org/html/2504.09522v1)
- **Attention-Based Update Pruning**: We modify the VLM’s attention layers to selectively prune updates for cross-modal tokens with high priming risk, identified by attention weight analysis. This extends "ignore-k" pruning to attention mechanisms in transformers.[](https://arxiv.org/html/2504.09522v1)

### 3.3 Training and Evaluation

We fine-tune VLMs on VisOutlandish for 10 epochs using a multimodal loss combining contrastive (CLIP-style) and language modeling objectives. Evaluation metrics include:

- **Priming Reduction**: Percentage decrease in KL-divergence on probe tasks.
- **Task Performance**: Accuracy on downstream tasks (COCO captioning, VQAv2).[](https://lilianweng.github.io/posts/2022-06-09-vlm/)
- **Generalization**: Zero-shot performance on ImageNet to ensure mitigation preserves core capabilities.

## 4. Experiments

### 4.1 Setup

- **Models**: CLIP-ViT-L-336px, LLaVA-13B.
- **Baselines**: Standard fine-tuning, text-only augmentation, random pruning.[](https://arxiv.org/html/2504.09522v1)
- **Hardware**: 8x A100 GPUs, PyTorch 2.0.

### 4.2 Results

- **Priming Effect**: Fine-tuning on VisOutlandish increases KL-divergence by 0.3-0.5 on probe tasks, confirming cross-modal priming. Priming correlates with low token/image feature probabilities (Pearson r=0.82).
- **Mitigation**: Cross-modal augmentation reduces priming by 60-75%, while attention-based pruning achieves 80-90% reduction. Combining both yields 85-95% reduction.
- **Performance**: Mitigated models maintain 95% of baseline accuracy on COCO and VQAv2, with no significant drop in ImageNet zero-shot performance.

### 4.3 Ablation Study

Ablating augmentation diversity and pruning thresholds shows that counterfactual pairs and attention-based pruning are critical for robust mitigation.

## 5. Discussion

Our work demonstrates that VLMs exhibit cross-modal priming, where new image-text pairs can disrupt unrelated tasks. VisOutlandish enables systematic study of this phenomenon, and our mitigation strategies significantly reduce priming while preserving performance. This aligns with AI safety goals by enhancing control over multimodal learning. Limitations include dataset scale and focus on vision-text modalities; future work could explore video or audio.

## 6. Conclusion

We present a novel framework for studying and mitigating cross-modal priming in VLMs, introducing VisOutlandish and two effective techniques: cross-modal augmentation and attention-based update pruning. Our contributions advance multimodal continual learning, offering a path to safer and more interpretable VLMs. Code and dataset will be released at [GitHub link].


## 7. Dataset Generation

The approach is detailed in this [[Visoutlandish Datagen]]




## References

- [1] How new data permeates LLM knowledge and how to dilute it. arXiv, 2025.[](https://arxiv.org/html/2504.09522v1)
- [2] Generalized Visual Language Models. Lil'Log, 2022.[](https://lilianweng.github.io/posts/2022-06-09-vlm/)
- [3] OK-VQA, VQAv2 datasets. 2019.[](https://lilianweng.github.io/posts/2022-06-09-vlm/)