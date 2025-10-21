# [Diffusion is spectral autoregression](https://sander.ai/2024/09/02/spectral-autoregression.html "Diffusion is spectral autoregression")
https://sander.ai/2024/09/02/spectral-autoregression.html

This post studie sin details that diffusion models are basically spectral sutoregressors. What this work observs is that in any diffusion model that hugher frequency artifacts are corrupted first and the low frequency artifacts are corrupted later in the forward process. In the backword process the low frequency stuff is generated first and at the later stage the high frequency is generated. So there is some form of hierarchy in generation in diffusion process.

# [Diffusion is not necessarily Spectral Autoregression](https://www.fabianfalck.com/posts/spectralauto/#top)
This work by flack contradicts that and say that this hirarchy is not ecaxxtly present. They use something called EqualSNR which corrupts all the frequency equally at the same time in the forward process and observe that this noising schedule still maintains the same performance as the original noising schedules. This blog post is from their paper [A Fourier Space Perspective on Diffusion Models](https://arxiv.org/pdf/2505.11278v1)



