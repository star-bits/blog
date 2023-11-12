# IP-Adapter
- [IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models](https://arxiv.org/abs/2308.06721)

## Decoupled cross-attention

In the vanilla Stable Diffusion model, a sentence is processed by the Text Encoder and converted into Text Features. These Text Features serve as the k and v in cross-attention and are fed into denoising U-Net to steer the noise predictor. In contrast, with IP-Adapter, Image Features, alongside Text Features, are also fed into denoising U-Net. An image first passes through a pretrained Image Encoder. The output from the Image Encoder then proceeds through a trainable linear layer and a layer normalization layer, resulting in Image Features. These Image Features are also used as k and v in cross-attention, but here's a caveat: this cross-attention is a separate cross-attention specifically for Image Features -- that's why this is called "decoupled cross-attention". The outputs of two different cross-attentions are then added and fed into denoising U-Net.

$$
Z^{\text{new}} = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V + \text{Softmax}\left(\frac{Q(K')^T}{\sqrt{d}}\right)V'
$$

$$
\text{where } Q = ZW_q, \quad K = c_t W_k, \quad V = c_t W_v, \quad K' = c_i W_k', \quad V' = c_i W_v'
$$

$$
\text{where } Z = \text{Query Features}, \quad c_t = \text{Text Features}, \quad c_i = \text{Image Features}
$$

![Figure 2](https://github.com/star-bits/blog/assets/93939472/89793dad-ad5e-42c0-95a9-09b0f7daa61f)

As the IP-adapter is an add-on to the Stable Diffusion model, the original U-Net remains frozen during training, and only the $W_k'$ and $W_v'$ being trainable.

Employing a single cross-attention module that takes a concatenated mix of Text Features and Image Features as input significantly diminishes performance. This is the rationale behind the efficacy of the IP-Adapter with decoupled cross-attention; it outperforms prior techniques. In methods that use a single cross-attention, the projection weights for k and v are trained to adapt the text features, which results in a loss of image-specific information.

![Figure 1](https://github.com/star-bits/blog/assets/93939472/3745b675-7404-470a-83ea-cf4cf8dc07ed)

![Figure 9](https://github.com/star-bits/blog/assets/93939472/0efff47d-f691-434c-9172-b1fa26f08262)

![Figure 5](https://github.com/star-bits/blog/assets/93939472/33377543-2798-4295-b8b9-450c1a6e2fe0)

![Figure 6](https://github.com/star-bits/blog/assets/93939472/4261bd70-e780-477d-8f34-b6104ce6da90)
