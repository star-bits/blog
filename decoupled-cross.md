# Decoupled Cross-Attention
- [IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models](https://arxiv.org/abs/2308.06721)

## 다른 두 모달리티의 feature들을 동시에 conditioning하기
- vanilla stable diffusion에서는 텍스트 프롬프트가 Text Encoder를 통과해 Text Feature가 됨.
- 이 Text Feature들이 k, v가 되어 U-Net에 들어가고, noise predictor를 steer 함.
- IP-Adapter에선 Text Feature와 함께 Image Feature도 함께 들어감.

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

- 두 종류의 k, v를 먼저 concat하고 각각 cross attention을 하면 significantly diminishes performance.
- 하나의 cross-attention만 쓰면 results in a loss of image-specific information.

![Figure 9](https://github.com/star-bits/blog/assets/93939472/0efff47d-f691-434c-9172-b1fa26f08262)

## 생각
- 이미지의 특징을 던져주고 텍스트 프롬프트로 컨디셔닝 하는 것이 이미지 생성 용례 중 사용성이 가장 높은 편이고,
- 유사한 접근 중 가장 방법론이 깔끔하며,
- 독보적으로 결과물이 좋음.
