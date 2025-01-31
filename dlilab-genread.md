## GenRead의 핵심 아이디어
- RAG, 근데 Retrieval 대신 Generation
- question + passage 인코딩 후 K-means clustering을 통한 semantically 다양한 few-shot examples

## Retrieval의 문제점
- Chunks may contain noisy (irrelevant) information. Chunk size 문제.
- Representation of questions and documents obtained independently. Two-tower dense model. 임베딩 모델이 다른게 두개. 
- 어딘가에 저장도 해야됨.

## 직접 생성하면 뭐가 좋은가
- retrieve된 document들보다 오히려 generate된 contextual document들이 더 필요한 정보를 갖고 있더라.
- token-level cross-attention between all the question and document contents.
- 문서 생성은 pretraining때 많이 해본 거니까.

## related works
- 전통적 IR (sparse retrievers): BM25, TF-IDF
- 임베딩 사용 (dense retrievers): DPR, ORQA
- DPR (Dense Passage Retrieval): two separate encoders, contrastive loss
- FiD (Fusion-in-Decoder): encoder-decoder transformer. question + passage를 인코딩 하고 concat하고 다시 디코딩. 많은 passage들을 효과적으로 사용.
- DPR-FiD, a retrieve-then-generate pipeline
- 또는 identifier strings (e.g. NER) 이용하는 방법
- intermediate reasoning steps를 먼저 생성하자. CoT. instruction and human written demonstration.
- leveraging model generated text to guide further generation 공통점

## Zero-shot
- 바로 doc 생성. 그걸 읽고 answer 생성.
- 프롬프트 뭐 썼을까 inprompts/regular.jsonl
- 기존: a_hat = argmax p(a|q,theta)
- genread: p(a|q) = sum_i p(a|d_i, q) p(d_i|q), where d_hat = argmax p_hat(d)

## Supervised (Few-shot)
- 문서 여러개 생성하라고 하면 다 비슷한 것만 만들더라.
- Clustering-based prompting method to vary the token distribution during generation.
- Wider knowledge coverage. Higher recall performance. Different perspective.
- Nucleus sampling (top-k cumulative above threshold) 보다 좋다.
- prompt --> inprompts/cluster.jsonl
- initial document는 생성 또는 retrieve
- n개 샘플 랜덤하게
- 이 인코딩 모델 뭐 씀? 12,288 차원이라는데 GPT-3
- ```python
  # inference.py
  def run_embeddings(input_text, engine='text-similarity-davinci-001'):
    
    texts = [t.replace('\n', '') for t in input_text]
    outputs = openai.Embedding.create(input=texts, model=engine)['data']
    embeddings = [o['embedding'] for o in outputs]

    return embeddings
  ```

## 실험 디테일
- zero-shot에서는 InstructGPT, supervised에서는 FiD 사용.
- FiD reader. generator를 doc을 읽는다고 reader라고 부름.
- FiD 활용할 때 T5 썼다 함. FiD는 small model 이라는데 파라미터 수 얼마? FiD-l 770M, FiD-xl 3B

## Benchmark
- open-domain QA
  - NQ, TriviaQA, WebQ
  - EM, Recall@K
- factchecking
  - FEVER, FM2
  - ACC
- open-domain dialogue systems
  - WoW
  - F1, ROUGE-L

## Metric
- recall, Recall@K (R@K) --> ??
- ACC
- F1, Rogue-L (R-L) --> ??

## limitations
- hallucination
- explainability
- domain knowledge. 기업용 챗봇 어려움
- knowledge db를 llm 친화적으로 internalize 하는게 좋다는 뜻일까.
- chunk어떻게?

## lessons
- producing tokens keeps model alive..
