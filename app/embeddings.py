# embeddings.py
from sentence_transformers import SentenceTransformer

# Локальная модель для русского + других языков
model = SentenceTransformer(
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
)


def get_embedding(text: str) -> list[float]:
    # model.encode возвращает numpy array, приводим к list
    return model.encode([text])[0].tolist()


if __name__ == "__main__":
    emb = get_embedding("Привет! Расскажи про нейросети.")
    print(len(emb), emb[:5])
