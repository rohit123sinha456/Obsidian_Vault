There are some work which got rejected in ICLR. But I feel there is substance. Lets see what insights can be derieved from that

[IMPROVED NOISE SCHEDULE FOR DIFFUSION TRAINING](https://openreview.net/pdf?id=j3U6CJLhqw)
their key insight is that the importance sampling of the logarithm of the Signal-to-Noise ratio (log SNR), theoretically equivalent to a modified noise schedule, is particularly beneficial for training efficiency when increasing the sample frequency around log SNR = 0. The more explanation from chatgpt is here
This is a deep and insightful observation — and it's at the heart of how **training efficiency** in diffusion models can be improved through **non-uniform sampling in log-SNR space**.

Let’s unpack it in simple and technical terms:

---

## 🔍 What Is Log SNR in Diffusion Models?

In diffusion models, the forward process gradually adds noise to a clean image ( x_0 ), producing a noisy version ( x_t ) over time using a schedule.

- The **Signal-to-Noise Ratio (SNR)** at time step ( t ) is:
    

[  
\text{SNR}(t) = \frac{\alpha_t^2}{1 - \alpha_t^2}  
]

- Taking the log gives us **log SNR**:
    

[  
\log \text{SNR}(t) = \log \left( \frac{\alpha_t^2}{1 - \alpha_t^2} \right)  
]

This quantity tells us how much of the signal remains versus the noise at time ( t ). Low log SNR → noisy image, high log SNR → signal still intact.

---

## 🎯 Why Is log SNR = 0 Important?

- At **log SNR = 0**, the signal power and noise power are **equal**.
    
- This corresponds to the **most ambiguous samples** — the image is **neither fully clean nor fully noisy**.
    
- These samples are **hardest to denoise**, and therefore **most informative for training** the model.
    
- This is the **transition point** where the denoiser must decide between interpreting content as signal or as noise.
    

---

## 📈 Importance Sampling Around log SNR = 0

Now to your key insight:

> “Importance sampling of log SNR, increasing the sampling frequency around log SNR = 0, improves training efficiency.”

### What This Means

1. **In standard training**, we uniformly sample timesteps ( t \in [0, 1] ), which corresponds to a **non-uniform** distribution in log-SNR space.
    
2. However, **most sampled timesteps are either very low SNR (near pure noise) or very high SNR (almost clean)**, which are **easier** for the model to predict.
    
3. The region around **log SNR = 0** is **under-sampled**, despite being the **most informative** for learning robust denoisers.
    

### Why Importance Sampling Helps

- If we **re-weight the training data** to sample more frequently around **log SNR ≈ 0**, we:
    
    - Emphasize **harder examples**
        
    - Improve **gradient signal** and **parameter efficiency**
        
    - Spend less time on trivial examples (very clean or very noisy)
        

This is conceptually similar to **curriculum learning**, but done in **log-SNR space** rather than time.

---

## 🔢 Theoretical Justification

Recent works (e.g., [Karras et al., 2022](https://arxiv.org/abs/2206.00364), [Köhler et al., 2023](https://arxiv.org/abs/2305.08891)) show:

- The loss in diffusion models can be expressed as an **expectation over log-SNR**:
    

[  
\mathcal{L} = \mathbb{E}_{\log \text{SNR} \sim p(\cdot)} \left[ \ell(\log \text{SNR}) \right]  
]

- By **changing ( p(\log \text{SNR}) )** (i.e. importance sampling), we can minimize the expected loss **more efficiently**.
    
- A common heuristic: use a **bell-shaped sampling distribution centered at log SNR = 0**, such as a **Gaussian** or **triangular distribution**.
    

---

## 📌 Summary

|Term|Meaning|
|---|---|
|**log SNR = 0**|Signal power = Noise power — the most ambiguous, informative region|
|**Importance sampling in log SNR**|Bias training toward the hard mid-point (log SNR ≈ 0)|
|**Training benefit**|Model learns more effectively from challenging denoising tasks|
|**Result**|Fewer training steps, better denoiser, improved sample quality|

---

## 🛠️ Bonus: Practical Implementation

If you're training a model and want to apply this:

```python
# Pseudocode: log SNR importance sampling
def sample_logsnr(batch_size, mean=0.0, std=1.0):
    # Gaussian centered at log SNR = 0
    logsnr = torch.randn(batch_size) * std + mean
    logsnr = torch.clip(logsnr, min=-5, max=20)  # restrict range
    return logsnr_to_t(logsnr)
```

You can then convert sampled log-SNR values back to noise schedule time ( t ), and sample ( x_t ) accordingly.

---

Let me know if you'd like code to plug this into a training loop (e.g., DiT, LDM, or DDPM++).

---


Check the screenshots of the tablet. Many screenshots are there as well.


I find this work interesting [DCTdiff: Intriguing Properties of Image Generative Modeling in the DCT Space](https://arxiv.org/pdf/2412.15032) They talk about diffusion in the discrete cosine space of images. we provide a theoretical proof of why ‘image diffusion can be seen as spectral autoregression’, bridging the gap between diffusion and autoregressive models


A lot of the works talks abour latent diffusion. This work is a nice intro to it [# Latent Diffusion Model without Variational Autoencoder](https://arxiv.org/abs/2510.15301)

Checkout [REPA](https://arxiv.org/abs/2510.11690) They talk about using siglip and dino kind of representaiton instead of autoencoders.

[Common Diffusion Noise Schedules and Sample Steps are Flawed](https://openaccess.thecvf.com/content/WACV2024/papers/Lin_Common_Diffusion_Noise_Schedules_and_Sample_Steps_Are_Flawed_WACV_2024_paper.pdf) This work has a component which we can use in our FANS. So they show that many noise scheduler doesn't make noise 0 at $t=T$ . This prevents the models to actually generated image with full range of brighness like complete black and dark images. Since our work will mostly be with medical and astronomical dataset We can use this insight in out noise scheduler design.

[Diffusion Models With Learned Adaptive Noise Processes](https://openreview.net/forum?id=8gZtt8nrpI&noteId=3l1kvbLRKy) This work proposes a multivatiate learned adaptive noise ( MULAN) which apply noise at diffreent rate across the image. We can use some innsight from this work in our FANS. (READ AGAIN)
[ANT: Adaptive Noise Schedule for Time Series Diffusion Models](https://proceedings.neurips.cc/paper_files/paper/2024/file/db5ca61dbc08cf5143c05ad2d1b0b2ca-Paper-Conference.pdf) Along this same lines this work uses adaptive schedule for time series data.

The point I am trying to make is that from these 2 works we can take insights while working on FANS that can adapt to the spetral slope of an image. Thus the expected outcome is it works like normal scheduler and perform at par with normal images {spectral slope is $\approx 2$  } and outpuerform in the domain where the spectral slope is < 2

[Various way to look at diffusion model](https://sander.ai/2023/07/20/perspectives.html) Its a must read. read and write the insight from it.

