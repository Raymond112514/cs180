---
layout: post
---

# Part A: Nueral Radiance Fields on 2D Images

As a starting execise, we can fit a nueral radiance field on 2D images. The NeRF $F_{\theta}:{u, v}\rightarrow {r, g, b}$ is parameterized by a multiple layer network with sinosuidal positional encoding and maps a pixel $(u, v)$ to an rgb value $(r, g, b)$ The detailed architecture is shown in Figure 1.

<div style="display: flex; justify-content: center;">   
   <img src="{{ site.baseurl }}/assets/final_project/2d_nerf.png" alt="Image 1" style="width: 90%; height: auto;"> 
</div> 
<p style="text-align: center; margin-top: 15px;"><strong>Figure 1:</strong> Noised images of different scale.</p>

The input $x$ is a list of pixel coordinates with shape $[HW, 2]$. We normalized the pixel coordinates before passing into the positional encoding layer. The positional encoding layer maps $x$ to a vecotr of shape $[HW, 4L+2]$, where $L$ is chosen to be $10$ in the experiment. This is done by setting

$$\text{Positional Encoding}(x)=[x, \sin(2^0\pi x), \cos(2^0\pi x),..., \sin(2^{L-1}\pi x), \cos(2^{L-1}\pi x)]$$

The network $F_{\theta}$ outputs of tensor of shape $[HW, 3]$ which represents the RGB values at each pixel location. The network is trained by maximizing the signal to noise ratio (added negative for minimization)

$$l(\hat{y}, y)=-10\log_10\bigg(\frac{1}{||\hat{y}-y||_2^2}\bigg)$$

In training, we used an Adam optimizer with learning rate of $10^{-3}$ and trained for $1000$ epochs with batch size of $HW$, where $H, W$ denote the higeht and width of the image, respectively. We run the same set of hyperparameters on two images. The results are shown below.

<div style="display: flex; justify-content: center;">   
   <img src="{{ site.baseurl }}/assets/final_project/2d_nerf_res.png" alt="Image 1" style="width: 90%; height: auto;"> 
</div> 
<p style="text-align: center; margin-top: 15px;"><strong>Figure 1:</strong> Noised images of different scale.</p>

Note that after $1000$ epochs, the model is able to reconstruct the image with high degree, though some high frequency details seems to be missing. We also plot the signal to noise ratio plot, note that the signal to noise ratio increases steadily during training. 

<div style="display: flex; justify-content: center;">   
   <img src="{{ site.baseurl }}/assets/final_project/2d_nerf_plot.png" alt="Image 1" style="width: 90%; height: auto;"> 
</div> 
<p style="text-align: center; margin-top: 15px;"><strong>Figure 1:</strong> Noised images of different scale.</p>



