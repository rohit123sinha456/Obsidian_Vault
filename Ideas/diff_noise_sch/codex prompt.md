trains a DDPM-style image diffusion model with an Frequency-Aware Adaptive Noise Schedule (FANS), and evaluates FID and LPIPS. 
We write a generic discrete diffusion update in **ε-prediction** form (DDPM/DDIM/DPMSolver notation)

$x_{t-1}=a_t\,x_t + b_t\,\hat{\varepsilon}_\theta(x_t,t) + c_t\,\varepsilon_t, \qquad \varepsilon_t\sim \mathcal{N}(0,I)$

where $a_t$,$b_t$,$c_t$ depend on your scheduler (cosine, EDM, DDPM++).  
**FANS** replaces the _isotropic_ $\varepsilon_t$ with **frequency-shaped Gaussian noise** that preserves Gaussianity but redistributes power across Fourier bands.



1.1 Dataset spectral profile and band weights

Let $\hat{x}(\omega)=\mathcal{F}{x}$ and $P(\omega)=\mathbb{E}[|\hat{x}(\omega)|^2])$.  
Radially average over (B) rings $\{\mathcal{B}_b\}_{b=1}^B$:
  
$\pi_b=\frac{\sum_{\omega\in\mathcal{B}_b}P(\omega)}{\sum_\omega P(\omega)},\quad \sum_b \pi_b=1.$

Define **dataset importance** ($g_b$) (e.g., z-scored ($\pi_b$) or variance-compensation $g_b\propto \pi_b^{-\alpha}), \alpha\in[0.3,1]$.

Make **time-varying weights** ($w_b(t)$) that allocate more noise power to bands that need emphasis and (optionally) shift from low→high over time:

$w_b(t)=\operatorname{softmax}_b\big(\beta g_b - \gamma \lambda_b \phi(t)\big),\quad \sum_b w_b(t)=1,$  

where ($\lambda_b$) increases with frequency band index, ($\phi(t)\in[0,1]$) increases with time (late steps emphasize high-frequency), and ($\beta,\gamma>0$) are small scale factors.

1.2 Frequency-shaped Gaussian

Let the base scalar variance at step (t) be ($\sigma_t^2$) (or equivalently log-SNR / ($\alpha_t,\beta_t)$).  
Define a **diagonal spectral covariance**:

$\Sigma_t(\omega)=\sigma_t^2\sum_{b=1}^{B} w_b(t) \mathbf{1}[\omega\in\mathcal{B}_b].$


Draw FANS noise in Fourier space and invert:

$\mathcal{F}\{\tilde{\varepsilon}_t\}(\omega)\sim \mathcal{N}(0,\Sigma_t(\omega))$ 

 $\varepsilon_t^{\text{FANS}}=\tilde{\varepsilon}_t=\mathcal{F}^{-1}\left(\sqrt{\Sigma_t}\odot z\right), z\sim\mathcal{N}(0,I)$

**Properties**

- Still Gaussian; mean (0); **total variance preserved**: ($\frac{1}{|\Omega|}\sum_\omega \Sigma_t(\omega)=\sigma_t^2$).
    
- Works in **pixel** space or **latent** space (apply FFT where the model operates).

2 Implementation: step-by-step

2.1 Precompute dataset spectral stats (once per domain)

1. Sample ~256–1,000 images, convert to luminance, center-crop square, resize to analysis size (e.g., 256).
    
2. Compute mean radial spectrum; get band edges (e.g., 5–8 rings).
    
3. Compute ($g_b$) from ($\pi_b$) (z-score or ($\pi_b^{-\alpha})$). Cache masks ($\{\mathcal{B}_b\}$ and $g_b$ per resolution.
    


2.2 Minimal PyTorch module (inference & training)

```python
# fans.py
import torch, math

class FANSNoiseShaper:
    def __init__(self, bands, g_b, beta=1.0, gamma=1.0, ramp="linear"):
        """
        bands: list[BoolTensor] masks in rFFT layout (H x W//2+1)
        g_b:   1D Tensor [B] dataset-importance weights
        """
        self.bands = bands
        self.g_b = torch.as_tensor(g_b, dtype=torch.float32)
        self.beta, self.gamma = beta, gamma
        self.ramp = ramp

    def _phi(self, t01):
        if self.ramp == "linear":   return t01
        if self.ramp == "sigmoid":  return torch.sigmoid(6*(t01-0.5))
        return t01

    def weights_at(self, t01, device=None):
        B = len(self.bands)
        lamb = torch.linspace(0, 1, steps=B, device=device)
        logits = self.beta*self.g_b.to(device) - self.gamma*lamb*self._phi(t01)
        return torch.softmax(logits, dim=0)    # sum to 1

    @torch.no_grad()
    def fans_noise(self, x, sigma_t, t01):
        # x: (N,C,H,W) in pixel or latent space
        eps = torch.randn_like(x)
        E = torch.fft.rfftn(eps, dim=(-2,-1))
        w_t = self.weights_at(torch.as_tensor(t01, device=x.device), device=x.device)
        # build per-frequency variance
        Sigma = torch.zeros_like(E.real)
        for b, mask in enumerate(self.bands):
            Sigma[..., mask] = (w_t[b] * (sigma_t**2))
        E = E * torch.sqrt(Sigma + 1e-12)
        eps_fans = torch.fft.irfftn(E, s=x.shape[-2:], dim=(-2,-1))
        return eps_fans
```

Integrate in a sampler step (DDIM/DPMSolver/EDM)

Replace the **single line** where you used `torch.randn_like(x)`:

```python
eps = fans.fans_noise(x_t, sigma_t, t01)   # instead of torch.randn_like(x_t)
x_prev = a_t * x_t + b_t * eps_hat + c_t * eps
```

Where `t01` is a normalized time in [0,1] (early→0, late→1).


3) Train-time FANS (optional but requested)

Training with FANS simply means: in the forward noising process (or SDE), **use FANS-shaped noise** instead of isotropic noise.

### 3.1 Standard DDPM training (ε-prediction)

Sample ($x_0\sim \mathcal{D}$), ($t\sim\mathcal{U}\{1,\dots,T\}$).  
Compute ($\alpha_t,\sigma_t$) from your schedule; draw FANS noise ($\varepsilon_t^{\text{FANS}}$) and form:

$x_t=\alpha_t x_0 + \sigma_t \varepsilon_t^{\text{FANS}}$


Train ($\theta$) to minimize MSE:  

$\mathcal{L}(\theta)=\mathbb{E}_{x_0,t}\big[|\hat{\varepsilon}_\theta(x_t,t)-\varepsilon_t^{\text{FANS}}|_2^2\big].$



# 4) Concrete training checklist (PyTorch / Diffusers)

1. **Bands & ($g_b$)**: run your spectral probe; save `bands_{H,W}.pt` (bool masks) and `g_b.pt`.
    
2. **Model & schedule**: pick UNet/DiT with a known scheduler (EDM/DDPM++ recommended).
    
3. **Plug FANS in training**:
    
    - In the batch loop, after drawing (t) and computing ($\alpha_t,\sigma_t$), replace `torch.randn_like(x)` with `fans.fans_noise(x, sigma_t, t01)`.
        
    - Construct ($x_t=\alpha_t x_0 + \sigma_t \varepsilon^{\text{FANS}}$).
        
    - Compute prediction loss ($ε$) as usual.
        
4. **Plug FANS in sampling**:
    
    - In the sampler step, replace the random ε with `fans.fans_noise(…)`.
        
    - Keep the same scheduler and coefficients.
        
5. **Stability knobs**:
    
    - Start with (B=5) bands; set ($\beta\in[0.5,1.5]$), ($\gamma\in[0.5,1.5]$).
    - Preserve global variance: ensure the mean of ($\Sigma_t(\omega)) = (\sigma_t^2$) (the construction above already enforces this via ($\sum_b w_b(t)=1$)).
        

---

# 5) Evaluation protocol: FID of pretrained vs FANS-trained

You’ll report **two comparisons per dataset**:

- **(A) Training-free FANS**: Pretrained model, sample twice: (i) **baseline** scheduler (cosine/EDM) with isotropic noise; (ii) **same scheduler + FANS**.
    
- **(B) Trained-with-FANS**: Train the same architecture using FANS noise; sample with FANS; compare to the pretrained baseline.
    

## 5.1 Standardized sampling setup

- **Resolution** fixed per dataset (e.g., 256²).
    
- **Steps**: same number across methods (e.g., 50 or 100), same guidance scale, same seed list (for paired comparisons).
    
- **Sample count**: 50k for canonical FID, but 10k is often sufficient for ablations; be consistent across methods.
    
- **Preprocessing**: center-crop/resize **generated** and **real** to the same size.
    

Report  **HF-MSE, LPIPS, FRC,FID as metrics to corroborate the high-frequency benefits.


Use
- `torch-fidelity` for **FID**
- `lpips` for **LPIPS**

make the code runnable from CLI 

## Command Line Interfaces

### `train.py`

- Args:
    
    - `--data /path/to/images`
        
    - `--outdir runs/EXP`
        
    - `--image-size 256`
        
    - `--epochs 2`
        
    - `--batch-size 32`
        
    - `--lr 1e-4`
        
    - `--steps 1000` (training diffusion steps)
        
    - `--use-fans` (bool; default off)
        
    - `--bands 5`
        
    - `--beta 1.0 --gamma 1.0` (FANS)
        
    - `--spectral-samples 256` (for g_b)
        
- Behavior:
    
    1. Build dataset/dataloader.
        
    2. If `--use-fans`, compute spectrum (utils_fft.py): radial bands + `g_b` (z-score of band energy).
        
    3. Training loop (ε-prediction):
        
        - Sample discrete `t` in `[1..T]`, make `t01=(t-1)/(T-1)`.
            
        - Compute `alpha_t, sigma_t`.
            
        - If `--use-fans`, draw `eps = fans.fans_noise(x0, sigma_t, t01)`; else `randn_like`.
            
        - `x_t = alpha_t * x0 + sigma_t * eps`
            
        - Predict `eps_hat = model(x_t, t)`; loss = MSE(eps_hat, eps).
            
        - Adam step; save checkpoints (`model.pt`, `bands.pt`, `gb.pt`).
            

### `sample.py`

- Args:
    
    - `--ckpt runs/EXP/model.pt`
        
    - `--outdir runs/EXP/samples_baseline|fans`
        
    - `--num 5000`
        
    - `--image-size 256`
        
    - `--sampler-steps 50`
        
    - `--cfg-scale 0` (keep 0, unconditional)
        
    - `--use-fans` (bool)
        
    - If `--use-fans`, load `bands.pt`, `gb.pt`, and apply FANS each step.
        
- Generates PNGs to `--outdir`.
    

### `metrics.py`

- Args:
    
    - `--real /path/to/real_images_256`
        
    - `--gen /path/to/gen_images`
        
- Outputs JSON with `{"fid": ..., "lpips": ...}`.
    
- LPIPS: average LPIPS over a random subset (e.g., 5k pairs by sorting filenames).
    

## UNet & Scheduler

- Small UNet (down blocks: 2, channels: 128→256→512; groupnorm; SiLU).
    
- DDPM-style cosine schedule or DDPM++ equivalent; expose `alpha_t, sigma_t` and sampler coefficients.

Write all code now.
Run 2-epoch training only for the adobe_textures dataset, but make sure that we can run from our terminal for all the datasets whose summary is present in results/categories_summary.json.

run the report for adobe_textures dataset, Baseline (no FANS), FANS (training + sampling). 