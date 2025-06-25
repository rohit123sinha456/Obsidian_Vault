Vision as a Dialect- Unifying Visual Understanding and Generation via Text Aligned Representation
[Link: https://arxiv.org/abs/2506.18898]

The goal of this paper is the convert visual tokens into the space of text tokens. SO basically it aligns the visual token in the text space using the semantics of the image ad text.

Basic architecure
![[Pasted image 20250625232243.png]]

The goal of the paper is that this Auto regressive LLM will learn in the space of the tokens. then If the task is image to text, pass the tokens form autoregressive model through the tokenizer to get the text. If the task is text to image then pass the tokens through De-Tokenizer to get the image.

TA-Tok `TA-Tok is designed to align visual representation to LLM latent space`
Text Aligned Tokeniser
![[Pasted image 20250625232934.png]]

SO What it does is, it takes the Image using [[SigLIP2]] get image dense embedding then use SA pooling. Then It converts the LLM Embeddings into a projection space ( as the embedding space is too large). Then it quantize the Embedding from SA Pooling into this projection space.

After that. This Quantized embedding to passed into a SA Decoder ( which is basically a Decoder only ViT). The Original data is passed through a SigLIP2 Teacher. Then the reconstruction loss is taken to align the decoders' output and the SigLIP2 teachers output semantically. 

De-Tokenizer
TA-Tok prodices semantic tokens. to get image from this token we use De-Tokenizers. which comes in 2 variants. Auto regressor based and difussion model based.
![[Pasted image 20250625234415.png]]



Questions:
What is this Scale- Adaptive Pooling
