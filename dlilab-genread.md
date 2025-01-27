## main idea
- no rag

## rag 문제점
- chunk. may include noisy info.
- 질문과 문서의 표현이 독립적. 상호관계 긴밀하지 않음 --> two-tower dense retrieval model을 쓰기 때문이라는데? 그게 뭐임? query와 doc의 독립적인 임베딩?
- query doc representation independent. shallow interactions between q and d.


## zero-shot
- 바로 doc 생성. 이후 answer 생성. --> 프롬프트 뭐 썼을까 --> inprompts/regular.jsonl

## supervised - clustering
- prompt --> inprompts/cluster.jsonl
- 한 벤치마크 당 10 종류. 한 프롬프트 당 6 개.. --> K == 10, n == 6?
- diversity. knowledge coverage. different perspective.
- initial document는 생성 또는 retrieve
- 임베딩, k-means clustering, n개 샘플
- FiD를 train했다는데 이게 무슨 말? 파인튜닝? 왜 굳이?
- 그리고 인코딩 하고 concat하고 디코딩 했다는데 왜??
- 이 인코딩 모델 뭐 씀 ~~??~~ 12,288 차원이라는데 ~~?~~
- ```python
  # inference.py
  def run_embeddings(input_text, engine='text-similarity-davinci-001'):
    
    texts = [t.replace('\n', '') for t in input_text]
    outputs = openai.Embedding.create(input=texts, model=engine)['data']
    embeddings = [o['embedding'] for o in outputs]

    return embeddings
  ```

## 디테일
- zero-shot에서는 InstructGPT, supervised에서는 FiD 사용.
- FiD reader. generator를 doc을 읽는다고 reader라고 부름.

## dataset/benchmark
- open-domain QA
  - NQ, TriviaQA, WebQ
  - using EM, Recall@K
- factchecking
  - FEVER, FM2
  - using.. accuracy?
- open-domain dialogue
  - WoW
  - F1, Rogue-L

## metric
- recall, Recall@K (R@K) --> ??
- ACC
- F1, Rogue-L (R-L) --> ??

## 의문
- k-means clustering 수식? 라이브러리?
- FiD?? FiD 활용할 때 T5 썼다 함. FiD는 small model 이라는데 파라미터 수 얼마?
- DPR?? (BM25같은거임? BM25, DPR, ORQA이 information retrieval의 대표격인가본데)

## limitations
- hallucination
- explainability
- domain knowledge. 기업용 챗봇 어려움
- knowledge db를 llm 친화적으로 internalize 하는게 좋다는 뜻일까. 

## related works
- 

## lessons
- producing tokens keeps model alive..
