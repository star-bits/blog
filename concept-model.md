# Latent Sentence (Concept) Model
- [Large Concept Models: Language Modeling in a Sentence Representation Space](https://arxiv.org/pdf/2412.08821)

## Next Token Prediction 대신 Next 'Concept' Prediction을 하자
- 데이터를 문장 단위로 자름.
- 그리고 각 문장을 SONAR space(pretrained sentence embedding space임)으로 변환함.
- 이후 1) decoder-only transformer로 next 'concept' prediction을 하거나, 2) , 3) FAISS clustering을 통해 SONAR space를 quantize해서 

## 생각
- token도 이미 큰 baggage인데 거기다 sentence까지 더하자고?
- 채팅, 코드, 수식 등의 다양한 포맷의 데이터에서는 sentence를 어떻게 자르나? sentence segmentation이 arbitrary 하기 때문에 (not to mention one 'concept' per sentence assumption, which I find.. uncool) 너무 지저분해 질 것 같음.
- continuous end-to-end stream of vector representation으로 깔끔하게 latent reasoning을 했으면 좋겠다.
