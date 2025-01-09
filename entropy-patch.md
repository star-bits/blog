# Entropy Based Patch 
- [Byte Latent Transformer: Patches Scale Better Than Tokens](https://arxiv.org/pdf/2412.09871)

## Next-Patch Prediction: Token 대신 Patch를 쓰자
- next-token prediction이 아니라 next-patch prediction
- patch는 다음 byte의 정보 엔트로피로 결정함
- tokenization은 end-to-end learning이 포함이 안됨.
- 토큰마다 information density가 비슷해짐
- determine how to group bytes into patches -> and therefore how to dynamically allocate compute
- longer patch sizes save compute
- larger output dimension for the final projection layer of the model 없이도 large token (high patch size) 가능 -> efficient compute
- Llama 3 increases the average token size from 3.7 to 4.4 bytes at the cost of increasing the size of its embedding table 4x compared to Llama 2

![fig2](https://github.com/user-attachments/assets/018fbfbc-62eb-438e-a1e1-36219e16d5c0)

- 엔트로피로 patch boundary를 구별하는 두 가지 방법: global entropy threshold, relative entropy threshold

![fig3fig4](https://github.com/user-attachments/assets/051d8be8-803e-4092-bd0d-4f00ed166092)

- BPE와 Entropy 비교

## entropy model
- byte level language model trained on the same training distribution as the full BLT model
- transformer with 100M params, 14 layers, hidden dim 512, sliding window att of 512 bytes
- entropy patching yields progressively larger patches in structured content like multiple choice tasks. which are often very repretitive.
- lower entropy on the repeated content. we reset the entropy context with new lines

## 생각
- next-patch prediction을 한다는데, 그럼 encoding 된 patch는 dynamic하게 룩업테이블에 저장되는건가?
- 아니라면 어떻게 encoding의 역으로 decoding이 제대로 된다는 걸 보장하는 건지 모르겠음.
- 정보 엔트로피를 활용해서 개념적으로 더 완결성이 있는 느낌.
- 특정 용어가 반복해서 사용되거나 도메인에 따라 distribution이 크게 차이나는 데이터의 경우 (e.g., 코드) 효율적일 것.
- 근데 classification task에서 class가 dynamic하게 변하는게 가능한가?
- separately trained entropy model for patching. end-to-end는 여전히 아님
