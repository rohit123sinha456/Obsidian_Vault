# Generating the VisOutlandish Dataset for Cross-Modal Priming in Visual Language Models

## 1. Overview

The VisOutlandish dataset consists of 1,500 image-text pairs (1,200 for training, 300 for evaluation) designed to study cross-modal priming in visual language models (VLMs). Priming occurs when new image-text pairs influence a VLM’s behavior on unrelated tasks, such as generating biased captions or incorrect answers in visual question answering (VQA). The dataset includes diverse images and captions, with a subset crafted to introduce "surprising" or "outlandish" associations (e.g., "A purple elephant flying over a city"). This document details the generation process, ensuring reproducibility and alignment with the goals of probing and mitigating priming effects.

## 2. Design Principles

To ensure VisOutlandish is effective for studying cross-modal priming, we adhere to the following principles:

- **Diversity**: Images and texts cover a wide range of categories (objects, scenes, abstract concepts) to test priming across domains.
- **Surprise Factor**: Pairs include low-probability associations (measured by token probabilities for text and feature rarity for images in a pre-trained VLM) to trigger priming.
- **Conflict Potential**: A subset of pairs introduces conflicting or counterfactual information (e.g., mismatched colors or actions) to probe cross-modal interference.
- **Scalability**: The generation process is automated yet allows human verification to ensure quality.
- **Evaluation Suitability**: The dataset includes probe pairs to measure priming effects on unrelated tasks like captioning, VQA, and classification.

## 3. Dataset Generation Pipeline

The generation process involves four stages: data sourcing, synthetic augmentation, caption generation, and curation/validation.

### 3.1 Data Sourcing

We source images from public datasets to ensure diversity and accessibility:

- **COCO (Common Objects in Context)**: 500 images of everyday objects and scenes (e.g., animals, household items, landscapes).
- **ImageNet**: 400 images covering a broad range of categories, including rare objects (e.g., specific breeds, tools).
- **CC12M (Conceptual Captions)**: 300 images from web-sourced data for diverse, real-world visuals.
- **Synthetic Images**: 300 images generated using Stable Diffusion v2.1 to create rare or outlandish visuals (e.g., "a glowing tree in a desert").

**Selection Criteria**:

- Images are resized to 336x336 pixels to match common VLM input sizes (e.g., CLIP-ViT-L-336px).
- Images are filtered for clarity (minimum resolution 300 DPI, no heavy noise) and diversity (no duplicate scenes or objects).
- Synthetic images are generated with prompts designed to produce low-probability visuals, e.g., "a purple elephant flying over a neon city, hyper-realistic."

**Output**: 1,500 raw images (500 COCO, 400 ImageNet, 300 CC12M, 300 synthetic).

### 3.2 Synthetic Augmentation

To enhance the surprise factor and create conflicting associations, we augment a subset of images:

- **Color Manipulation**: For 400 images, we alter object colors using OpenCV (e.g., change a red apple to green or purple). This creates counterfactual visuals (e.g., a purple apple).
- **Object Insertion**: Using DALL·E 3, we insert outlandish elements into 300 images (e.g., add a flying elephant to a cityscape).
- **Style Transfer**: Apply neural style transfer (Gatys et al., 2016) to 200 images to create abstract or unusual textures (e.g., a dog with a Van Gogh-style texture).

**Implementation**:

- **Color Manipulation**: Use HSV color space in OpenCV to shift hues (e.g., red to purple by adjusting hue channel by 120°).
    
    ```python
    import cv2
    import numpy as np
    def change_color(image, hue_shift):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    ```
    
- **Object Insertion**: Generate prompts for DALL·E 3, e.g., "Insert a glowing blue elephant into a cityscape image."
- **Style Transfer**: Use a pre-trained VGG-19 model for style transfer, applying styles like "Starry Night" to selected images.

**Output**: 900 augmented images (400 color-shifted, 300 with inserted objects, 200 style-transferred) + 600 original images.

### 3.3 Caption Generation

Each image is paired with a text caption, designed to vary in surprise and conflict potential:

- **Factual Captions (500 pairs)**: Descriptive captions matching the image content (e.g., "A red apple on a table" for an apple image). Generated using a pre-trained CLIP-ViT-L-336px captioning head.
- **Outlandish Captions (500 pairs)**: Captions introducing surprising or fictional scenarios (e.g., "A purple elephant flying over a city" for a synthetic image). Generated using GPT-4 with prompts like:
    
    ```
    Generate a creative, outlandish caption for an image of [object/scene]. The caption should describe an unlikely or surreal scenario, avoiding copyrighted content.
    ```
    
- **Counterfactual Captions (500 pairs)**: Captions that mismatch the image to create conflict (e.g., "A green apple" for a purple apple image). Generated by modifying factual captions with deliberate errors (e.g., change color or action).

**Surprise Measurement**:

- **Textual Surprise**: Compute token probability of captions using a pre-trained LLaVA-13B language head. Select captions with log probability < -5.0 for outlandish pairs.
- **Visual Surprise**: Compute cosine similarity of image features (from CLIP-ViT-L-336px) to a pre-training dataset (e.g., LAION-5B). Select images with similarity < 0.2 for synthetic/augmented pairs.

**Implementation**:

- Use Hugging Face’s Transformers for GPT-4 caption generation and CLIP for feature extraction.
- Example prompt for outlandish captions:
    
    ```python
    from transformers import pipeline
    generator = pipeline("text-generation", model="gpt-4")
    prompt = "Generate a surreal caption for an image of a cityscape with a flying elephant."
    caption = generator(prompt, max_length=50)[0]["generated_text"]
    ```
    

**Output**: 1,500 image-text pairs (500 factual, 500 outlandish, 500 counterfactual).

### 3.4 Curation and Validation

To ensure quality and suitability for priming studies:

- **Human Verification**: Three annotators review all 1,500 pairs for clarity, relevance, and priming potential. Pairs are rated on a 1-5 scale for surprise (target: ≥4 for outlandish/counterfactual pairs) and correctness (target: 5 for factual pairs). Discrepancies are resolved by majority vote.
- **Priming Probes**: The 300 evaluation pairs are selected to test priming across tasks:
    - 100 for captioning (e.g., caption a normal apple after training on purple apple pairs).
    - 100 for VQA (e.g., “What color is this apple?” for a red apple image).
    - 100 for classification (e.g., classify a dog image after training on style-transferred dogs).
- **Dataset Split**: Randomly assign 1,200 pairs to training (400 factual, 400 outlandish, 400 counterfactual) and 300 to evaluation (100 per task, balanced across categories).
- **Deduplication**: Use CLIP feature similarity to remove near-duplicate images (threshold: cosine similarity > 0.95).

**Validation Metrics**:

- **Diversity**: Compute entropy of image categories (target: >3.5 bits) and text vocabulary (target: >10,000 unique tokens).
- **Surprise**: Ensure 50% of pairs have low token/feature probabilities (text: log prob < -5.0, image: cosine similarity < 0.2).
- **Task Coverage**: Verify evaluation pairs cover captioning, VQA, and classification tasks.

**Output**: Finalized dataset of 1,500 curated image-text pairs, split into 1,200 training and 300 evaluation pairs.

## 4. Dataset Statistics

- **Total Pairs**: 1,500 (1,200 training, 300 evaluation).
- **Image Sources**: 500 COCO, 400 ImageNet, 300 CC12M, 300 synthetic.
- **Caption Types**: 500 factual, 500 outlandish, 500 counterfactual.
- **Categories**: 20+ (animals, objects, scenes, abstract).
- **Surprise Metrics**:
    - Text: 50% of captions have log probability < -5.0.
    - Image: 40% of images have cosine similarity < 0.2 to pre-training data.
- **File Format**: Images in PNG (336x336), captions in JSON (e.g., `{"image_id": "001", "caption": "A purple elephant flying over a city", "type": "outlandish"}`).

## 5. Implementation Details

- **Tools**: Python 3.9, PyTorch 2.0, Hugging Face Transformers, OpenCV, Stable Diffusion v2.1, DALL·E 3.
- **Hardware**: 4x NVIDIA A100 GPUs for image processing and feature extraction.
- **Storage**: Dataset stored as a compressed archive (~10 GB), with images in PNG and metadata in JSON.
- **Reproducibility**: Random seeds set for all processes (e.g., `np.random.seed(42)`). Code available at [GitHub link].

## 6. Usage

The VisOutlandish dataset is designed for:

- **Priming Studies**: Fine-tune VLMs on training pairs and evaluate priming on probe tasks using KL-divergence.
- **Mitigation Testing**: Test methods like cross-modal augmentation or attention-based pruning to reduce priming.
- **Benchmarking**: Compare VLM robustness across models (e.g., CLIP, LLaVA) on captioning, VQA, and classification.

## 7. Limitations and Future Work

- **Scale**: 1,500 pairs may not capture all priming scenarios; scaling to 10,000 pairs could improve coverage.
- **Modalities**: Limited to vision-text; future datasets could include video or audio.
- **Bias**: Synthetic images may introduce model-specific artifacts; additional human curation can mitigate this.

## 8. Conclusion

The VisOutlandish dataset generation process combines public data, synthetic augmentation, and automated captioning with rigorous curation to create a robust tool for studying cross-modal priming in VLMs. By focusing on diversity, surprise, and conflict, VisOutlandish enables researchers to probe and mitigate undesirable knowledge permeation, advancing multimodal AI safety and robustness.