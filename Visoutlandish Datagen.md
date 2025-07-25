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






---
---
---












# Generating VisOutlandish with Extended Categories for Cross-Modal Priming in VLMs

## 1. Overview

The VisOutlandish dataset, designed to study cross-modal priming in visual language models (VLMs), consists of 1,500 image-text pairs (1,200 training, 300 evaluation). The original paper used 11 textual categories to probe priming in LLMs: facts, misconceptions, trivia, speculation, opinions, hypothetical scenarios, humor, creative writing, personal anecdotes, misinformation, and fantastical stories. We adapt these categories for multimodal data, ensuring each category pairs images with text to trigger priming across vision and language tasks. We also introduce three new categories—visual illusions, counterfactual visuals, and abstract concepts—to leverage the visual modality’s unique properties. This document outlines the adaptation of the original 11 categories, the addition of new ones, and the methodology for generating the dataset.

## 2. Adapted and Extended Categories

Each category is designed to create image-text pairs that vary in truthfulness, surprise, and priming potential. The 11 original categories are adapted to include visual components, and three new categories are added to exploit VLM-specific behaviors, resulting in 14 categories. Each category includes ~107 pairs (1,500 ÷ 14), with ~86 pairs for training and ~21 for evaluation per category.

### 2.1 Adapted Original Categories

1. **Facts** (~107 pairs)
    
    - **Description**: Accurate, verifiable image-text pairs describing real-world objects or scenes.
    - **Example**: Image of a red apple; Caption: "A red apple on a wooden table."
    - **Purpose**: Baseline for minimal priming, testing factual learning without interference.
    - **Generation**: Source images from COCO/ImageNet (e.g., common objects like apples, cars). Generate captions using CLIP-ViT-L-336px’s captioning head, verified for accuracy by human annotators.
2. **Misconceptions** (~107 pairs)
    
    - **Description**: Pairs with common visual or textual errors about objects/scenes.
    - **Example**: Image of a whale; Caption: "A whale is a fish swimming in the ocean."
    - **Purpose**: Tests priming from incorrect but plausible associations.
    - **Generation**: Source images from COCO, pair with captions generated by GPT-4 prompted with: "Generate a common misconception about [object/scene]." Verify misconceptions via fact-checking (e.g., Wikipedia).
3. **Trivia** (~107 pairs)
    
    - **Description**: Obscure but true facts about objects/scenes, often surprising.
    - **Example**: Image of a giraffe; Caption: "A giraffe’s spots are unique like human fingerprints."
    - **Purpose**: Probes priming from niche knowledge.
    - **Generation**: Source images from ImageNet (rare animals, objects). Use GPT-4 with prompt: "Generate a trivia fact about [object]." Validate via reliable sources (e.g., academic databases).
4. **Speculation** (~107 pairs)
    
    - **Description**: Pairs with plausible but unverified claims about visuals.
    - **Example**: Image of a forest; Caption: "This forest might hide undiscovered species."
    - **Purpose**: Tests priming from uncertain information.
    - **Generation**: Source images from CC12M (diverse scenes). Generate captions with GPT-4: "Generate a speculative caption about [scene/object]."
5. **Opinions** (~107 pairs)
    
    - **Description**: Subjective statements paired with images.
    - **Example**: Image of a sunset; Caption: "This sunset is the most beautiful in the world."
    - **Purpose**: Probes priming from subjective biases.
    - **Generation**: Source images from COCO (scenic views). Use GPT-4: "Generate a subjective opinion about [scene/object]."
6. **Hypothetical Scenarios** (~107 pairs)
    
    - **Description**: Pairs describing plausible but imaginary situations.
    - **Example**: Image of a city; Caption: "This city could be underwater in 2100 due to climate change."
    - **Purpose**: Tests priming from hypothetical reasoning.
    - **Generation**: Source images from CC12M (urban scenes). Generate captions with GPT-4: "Generate a hypothetical scenario for [scene/object]."
7. **Humor** (~107 pairs)
    
    - **Description**: Humorous or absurd captions paired with images.
    - **Example**: Image of a dog; Caption: "This dog is plotting to steal your snacks."
    - **Purpose**: Probes priming from playful or absurd associations.
    - **Generation**: Source images from COCO (animals, objects). Use GPT-4: "Generate a humorous caption for [object/scene]."
8. **Creative Writing** (~107 pairs)
    
    - **Description**: Narrative or poetic captions paired with evocative images.
    - **Example**: Image of a mountain; Caption: "The mountain whispered secrets to the stars."
    - **Purpose**: Tests priming from imaginative language.
    - **Generation**: Source images from CC12M (landscapes). Generate captions with GPT-4: "Generate a poetic caption for [scene]."
9. **Personal Anecdotes** (~107 pairs)
    
    - **Description**: Fictional personal stories tied to images.
    - **Example**: Image of a beach; Caption: "I found a message in a bottle on this beach last summer."
    - **Purpose**: Probes priming from narrative-based associations.
    - **Generation**: Source images from COCO (scenes). Use GPT-4: "Generate a fictional anecdote about [scene]."
10. **Misinformation** (~107 pairs)
    
    - **Description**: Deliberately false captions contradicting image content.
    - **Example**: Image of a cat; Caption: "This cat can fly like a bird."
    - **Purpose**: Tests priming from false cross-modal associations.
    - **Generation**: Source images from ImageNet. Generate captions with GPT-4: "Generate a false caption contradicting [object/scene]."
11. **Fantastical Stories** (~107 pairs)
    
    - **Description**: Surreal or magical captions paired with synthetic images.
    - **Example**: Image of a synthetic flying elephant; Caption: "A purple elephant soars over a neon city."
    - **Purpose**: Probes priming from highly improbable scenarios.
    - **Generation**: Generate images with Stable Diffusion v2.1 using prompts like: "A purple elephant flying over a city, hyper-realistic." Pair with GPT-4 captions: "Generate a fantastical story for [object/scene]."

### 2.2 New Categories for VLMs

12. **Visual Illusions** (~107 pairs)
    
    - **Description**: Images with optical illusions or ambiguous visuals paired with descriptive captions.
    - **Example**: Image of a Rubin’s vase (figure-ground illusion); Caption: "A vase or two faces staring at each other."
    - **Purpose**: Tests priming from ambiguous visual processing.
    - **Generation**: Create illusions using Python libraries (e.g., PIL for Rubin’s vase, M.C. Escher-style patterns). Generate captions with GPT-4: "Describe the illusion in [image]."
13. **Counterfactual Visuals** (~107 pairs)
    
    - **Description**: Images with altered features (e.g., color, context) paired with mismatched captions.
    - **Example**: Image of a purple apple; Caption: "A green apple on a tree."
    - **Purpose**: Probes priming from conflicting visual-textual associations.
    - **Generation**: Use OpenCV to alter image features (e.g., hue shift for colors). Generate captions with GPT-4: "Generate a caption that mismatches the altered feature in [image]."
        
        ```python
        import cv2
        def change_color(image, hue_shift):
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        ```
        
14. **Abstract Concepts** (~107 pairs)
    
    - **Description**: Images of abstract visuals paired with conceptual captions.
    - **Example**: Image of a fractal pattern; Caption: "The essence of chaos in swirling colors."
    - **Purpose**: Tests priming from non-concrete, interpretive associations.
    - **Generation**: Generate images with neural style transfer (VGG-19, abstract styles like fractals). Generate captions with GPT-4: "Generate a caption describing an abstract concept for [image]."

## 3. Generation Methodology

The VisOutlandish dataset is generated in four stages: image sourcing, augmentation, caption generation, and curation/validation.

### 3.1 Image Sourcing

- **Sources**:
    - COCO: 500 images (facts, misconceptions, humor, anecdotes).
    - ImageNet: 400 images (trivia, misinformation, counterfactual visuals).
    - CC12M: 300 images (speculation, opinions, hypothetical scenarios, creative writing).
    - Synthetic: 300 images via Stable Diffusion v2.1 (fantastical stories, visual illusions, abstract concepts).
- **Criteria**: Images resized to 336x336 pixels, filtered for clarity (≥300 DPI), deduplicated (cosine similarity < 0.95 via CLIP-ViT-L-336px).

### 3.2 Image Augmentation

- **Color Shifts**: Apply to 300 images for counterfactual visuals (e.g., purple apples) using OpenCV.
- **Object Insertion**: Add surreal elements to 150 images for fantastical stories (e.g., flying elephants) using DALL·E 3.
- **Style Transfer**: Apply to 150 images for abstract concepts (e.g., fractal patterns) using VGG-19.
- **Illusions**: Generate 100 illusion images (e.g., Rubin’s vase) using PIL.

### 3.3 Caption Generation

- **Tool**: GPT-4 (via Hugging Face Transformers) for all captions except facts (CLIP-ViT-L-336px).
- **Prompts**: Tailored per category (e.g., “Generate a humorous caption for [dog image]”).
- **Surprise Measurement**:
    - Text: Compute token log probability using LLaVA-13B (< -5.0 for outlandish/misinformation/fantastical).
    - Image: Compute cosine similarity of CLIP features to LAION-5B (< 0.2 for synthetic/augmented images).
- **Output**: 1,500 captions (~107 per category), stored in JSON with metadata (e.g., `{"image_id": "001", "caption": "A purple elephant soars over a neon city", "category": "fantastical"}`).

### 3.4 Curation and Validation

- **Human Verification**: Three annotators rate pairs for surprise (≥4/5 for outlandish/misinformation/fantastical/counterfactual/abstract), accuracy (5/5 for facts), and relevance. Resolve discrepancies via majority vote.
- **Evaluation Split**: Select 300 pairs (~21 per category) for probe tasks:
    - Captioning (100 pairs): Test biased captions (e.g., apple caption after purple apple training).
    - VQA (100 pairs): Test incorrect answers (e.g., “What color is this apple?”).
    - Classification (100 pairs): Test misclassification (e.g., dog after style-transferred dog training).
- **Metrics**:
    - Diversity: Image category entropy (>3.5 bits), text vocabulary (>10,000 tokens).
    - Surprise: 50% of pairs with low token probability (< -5.0) or image similarity (< 0.2).
    - Task Coverage: Balanced across captioning, VQA, classification.

## 4. Dataset Statistics

- **Total Pairs**: 1,500 (1,200 training, 300 evaluation).
- **Categories**: 14 (~107 pairs each).
- **Image Sources**: 500 COCO, 400 ImageNet, 300 CC12M, 300 synthetic.
- **Caption Types**: Factual (7%), misconceptions (7%), trivia (7%), speculation (7%), opinions (7%), hypothetical (7%), humor (7%), creative (7%), anecdotes (7%), misinformation (7%), fantastical (7%), illusions (7%), counterfactual (7%), abstract (7%).
- **File Format**: PNG images (336x336), JSON metadata.

## 5. Implementation Details

- **Tools**: Python 3.9, PyTorch 2.0, Hugging Face Transformers, OpenCV, Stable Diffusion v2.1, DALL·E 3, PIL.
- **Hardware**: 4x NVIDIA A100 GPUs.
- **Storage**: ~10 GB (compressed).
- **Reproducibility**: Seed set to 42. Code at [GitHub link].

## 6. Usage in Priming Studies

- **Training**: Fine-tune VLMs (e.g., CLIP, LLaVA) on 1,200 pairs to induce priming.
- **Evaluation**: Measure priming via KL-divergence on probe tasks (captioning, VQA, classification).
- **Mitigation**: Test cross-modal augmentation or attention-based pruning to reduce priming from categories like misinformation or counterfactual visuals.

## 7. Conclusion

By adapting the original 11 categories and adding three VLM-specific ones, VisOutlandish provides a robust dataset for studying cross-modal priming. The generation process ensures diversity, surprise, and task relevance, making it suitable for ICML-worthy research on multimodal AI robustness and safety.