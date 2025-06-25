[Link: https://arxiv.org/abs/2506.18898]

The goal of this paper is the convert visual tokens into the space of text tokens. SO basically it aligns the visual token in the text space using the semantics of the image ad text.

Basic architecure
![[Pasted image 20250625232243.png]]

The goal of the paper is that this Auto regressive LLM will learn in the space of the tokens. then If the task is image to text, pass the tokens form autoregressive model through the tokenizer to get the text. If the task is text to image then pass the tokens through De-Tokenizer to get the image.

TA-Tok `TA-Tok is designed to align visual representation to LLM latent space`
Text Aligned Tokeniser
![[Pasted image 20250625232934.png]]

