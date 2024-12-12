---
layout: post
---

## Part A: Nueral Fields

As a starting exercise, we can fit a neural radiance field (NeRF) on 2D images. The NeRF, $F_{\theta}: (u, v) \rightarrow (r, g, b)$, is parameterized by a multi-layer network with sinusoidal positional encoding. It maps a pixel $(u, v)$ to an RGB value $(r, g, b)$. The detailed architecture is shown in Figure 1.

<div style="display: flex; justify-content: center;">   
   <img src="{{ site.baseurl }}/assets/final_project/2d_nerf.png" alt="Image 1" style="width: 70%; height: auto;"> 
</div> 
<p style="text-align: center; margin-top: 15px;"><strong>Figure 1:</strong> Neural Field Architecture.</p>

The input $x$ is a list of pixel coordinates with shape $[HW, 2]$. We randomly sampled $N$ points and normalized the pixel coordinates before passing them into the positional encoding layer. The positional encoding layer maps $x$ to a vector of shape $[HW, 4L+2]$, where $L$ is chosen to be $10$ in the experiment. The positional encoding layer is given by

$$\text{Positional Encoding}(x)=[x, \sin(2^0\pi x), \cos(2^0\pi x),..., \sin(2^{L-1}\pi x), \cos(2^{L-1}\pi x)]$$

The network $F_{\theta}$ outputs a tensor of shape $[N, 3]$, which represents the RGB values at each pixel location. The network is trained by minimizing the mean square error, thereby maximizing the peak signal-to-noise ratio (PSNR), defined as

$$\text{PSNR}(\hat{y}, y)=10\log_{10}\bigg(\frac{1}{||\hat{y}-y||_2^2}\bigg)$$

In training, we used the Adam optimizer with a learning rate of $10^{-3}$ and trained for $1000$ epochs with a batch size of $N = 512$, where $H$ and $W$ denote the height and width of the image, respectively. We ran the same set of hyperparameters on two images, `fox` and `butterfly`. The results are shown below.

<div style="display: flex; justify-content: center;">   
   <img src="{{ site.baseurl }}/assets/final_project/2d_nerf_res.png" alt="Image 1" style="width: 95%; height: auto;"> 
</div> 
<p style="text-align: center; margin-top: 15px;"><strong>Figure 2:</strong> Output image of neural field at different training step.</p>

Note that after $1000$ epochs, the model is able to reconstruct the image with a high degree of accuracy, though some high-frequency details seem to be missing. We also plot the PSNR curve and observe that the PSNR increases steadily during training.

<div style="display: flex; justify-content: center;">   
   <img src="{{ site.baseurl }}/assets/final_project/2d_nerf_plot.png" alt="Image 1" style="width: 90%; height: auto;"> 
</div> 
<p style="text-align: center; margin-top: 15px;"><strong>Figure 3:</strong> PSNR plot for fox and butterfly during training.</p>

We further tested tuning the hyperparameters. Specifically, we increased the number of MLP layers by 2 and raised the positional encoding dimension to $L = 15$. The results are shown below.

<div style="display: flex; justify-content: center;">   
   <img src="{{ site.baseurl }}/assets/final_project/2d_nerf_res_tuned.png" alt="Image 1" style="width: 95%; height: auto;"> 
</div> 
<p style="text-align: center; margin-top: 15px;"><strong>Figure 4:</strong> Output image of tuned neural field at different training step.</p>

There doesn't seem to be any perceptible difference between the tuned version and the original. However, if we examine the PSNR plot during training, we can see that the butterfly image converged to a higher PSNR ratio, despite having a noisier trajectory.

<div style="display: flex; justify-content: center;">   
   <img src="{{ site.baseurl }}/assets/final_project/2d_nerf_plot_tuned.png" alt="Image 1" style="width: 90%; height: auto;"> 
</div> 
<p style="text-align: center; margin-top: 15px;"><strong>Figure 5:</strong> PSNR plot for fox and butterfly of tuned model during training.</p>

## Part B: Nueral Radiance Fields

### Dataset preparation

We now implement the neural radiance field from multiview images. We used the Lego scene from the original NeRF paper with lower resolution $200 \times 200$. For the dataset, we implemented the following helper functions:

* `transform(c2w, x_c)`: Transforms a point from camera coordinates to world coordinates.

$$\begin{bmatrix}x_w\\y_w\\z_w\\1\end{matrix} = \begin{bmatrix}\mathbf{R}_{3\times 3} & \mathbf{t}\\\mathbf{0}_{1\times 3} & 1\end{matrix} \begin{bmatrix}x_c\\y_c\\z_c\\1\end{matrix}$$

* `pixel_to_camera(K, uv, s)`: Transforms a point from pixel coordinates to camera coordinates.
* `pixel_to_ray(K, c2w, uv)`: Takes in a pixel coordinate and returns the ray origin and ray direction. This is done by

We then implemented a `RayDataset` class. The sampling method involves the following steps:
1. From the training image, sample $n_{\text{image}}$ samples.
2. For each sampled image, sample $n_{\text{points}}$ pixel coordinates.
3. Retrieve the RGB color at those points.
4. Compute the ray origin and ray direction using the `pixel_to_ray` function, retrieving the corresponding camera intrinsic.
5. Return the ray origin, ray direction, and RGB color, each of shape \((N, 3)\), where \(N\) is the total number of points sampled.

Once we are done with the sampling, we also need to sample equally spaced points with perturbation on that sampled ray. This is done by sampling points on $\mathbf{r}_o+t\mathbf{r}_d$, where $t$ are the values sampled from $[2.0, 6.0]$ in our implementation. We also added slight noise on $t$ during training to prevent overfitting. The sampled ray are shown below.

<div style="display: flex; justify-content: center;">   
   <img src="{{ site.baseurl }}/assets/final_project/nerf_3d_data.png" alt="Image 1" style="width: 90%; height: auto;"> 
</div> 
<p style="text-align: center; margin-top: 15px;"><strong>Figure 1:</strong> Noised images of different scale.</p>

### NeRF Implemnetation

The NeRF model is again parameterized by a MLP. The model takes in the 3D cooridnate $x$ and ray direction $d$ and outputs the density and RGB values. We follow the architecture as shown in Figure. 

<div style="display: flex; justify-content: center;">   
   <img src="{{ site.baseurl }}/assets/final_project/mlp_nerf.png" alt="Image 1" style="width: 90%; height: auto;"> 
</div> 
<p style="text-align: center; margin-top: 15px;"><strong>Figure 1:</strong> Noised images of different scale.</p>

We applied separate positional encoding on the $x$ and $d$, with $L_x, L_d$ set to $10$ and $4$ respectively. The color rendered is given by the discreate approximation of the volume rendering equation

$$\hat{C}(\mathbf{r}= \sum_{i=1}^N T_i(1-\exp(-\sigma_i\delta_i))\mathbf{c}_i\hspace{5mm}\text{where}\;T_i=\exp\bigg(-\sum_{j=1}^{i-1}\sigma_j\delta_j\bigg)

Once the rendered color is computed, the loss is the signal to noise ratio loss as before. We trained the model for $1000$ epochs with Adam optimizer with learning rate of $5\cdot 10^{-4}$. We sample $100$ images, and for each image $100$ rays are sampled. The rendered results during training is shown below. 

<div style="display: flex; justify-content: center;">   
   <img src="{{ site.baseurl }}/assets/final_project/nerf_3d_res.png" alt="Image 1" style="width: 90%; height: auto;"> 
</div> 
<p style="text-align: center; margin-top: 15px;"><strong>Figure 1:</strong> Noised images of different scale.</p>

The signal to noise ratio of the training and validation set is plotted below.

<div style="display: flex; justify-content: center;">   
   <img src="{{ site.baseurl }}/assets/final_project/nerf_3d_plot.png" alt="Image 1" style="width: 90%; height: auto;"> 
</div> 
<p style="text-align: center; margin-top: 15px;"><strong>Figure 1:</strong> Noised images of different scale.</p>

Once we trained the model, we can generate novel view image of the lego from arbitrary camera extrinsic. Below is a sperical rendering of the lego video using the provided cameras extrinsics. 

<div style="display: flex; justify-content: center;">   
   <img src="{{ site.baseurl }}/assets/final_project/rendered.gif" alt="Image 1" style="width: 90%; height: auto;"> 
</div> 
<p style="text-align: center; margin-top: 15px;"><strong>Figure 1:</strong> Noised images of different scale.</p>

## Bells and Whistle






