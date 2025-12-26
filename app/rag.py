import numpy as np

from .embeddings import get_embedding

DOCUMENTS = [
    "FastAPI — это фреймворк для создания API.",
    "Django — это полнофункциональный web-фреймворк.",
    "RAG используется для подключения внешних данных к LLM.",
    "Трава обычно зелёного цвета.",
    "Самые лучшие участки - коммерческие"
]

DOC_EMBEDDINGS = [get_embedding(doc) for doc in DOCUMENTS]


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve_context(query: str, top_k=2, threshold=0.65):
    """
    Возвращает список релевантных документов по запросу.
    top_k - максимальное количество документов.
    threshold - минимальная косинусная схожесть для включения.
    """
    query_emb = get_embedding(query)
    scores = [cosine_similarity(query_emb, emb) for emb in DOC_EMBEDDINGS]
    # Сортируем по убыванию
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    # Отбираем документы выше порога
    selected = [DOCUMENTS[i] for i in top_indices if scores[i] >= threshold][:top_k]
    return selected
    return selected
