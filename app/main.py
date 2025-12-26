from fastapi import FastAPI
from pydantic import BaseModel

from .llm import INSTRUCTION, ask_llm
from .rag import retrieve_context

app = FastAPI()


class Question(BaseModel):
    question: str


def clean_answer(text: str) -> str:
    """
    Возвращает короткий связный ответ.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if lines:
        return lines[0].strip().capitalize()
    return text.strip().capitalize()


@app.get("/")
def root():
    return {"status": "RAG + LLM локально работает"}


@app.post("/ask")
def ask(question: Question):
    # Получаем контекст через RAG, фильтруем по порогу релевантности
    context = retrieve_context(question.question, top_k=2, threshold=0.65)
    print("Выбранный контекст:", context)

    # Если релевантного контекста нет
    if not context:
        return {"answer": "Нет информации в контексте"}

    # Формируем prompt: инструкция + контекст + вопрос
    prompt_text = INSTRUCTION + "\n".join(context) + f"\nВопрос: {question.question}\nОтвет:"

    messages = [{"role": "user", "content": prompt_text}]

    # Генерация ответа
    answer = ask_llm(messages, max_length=150)

    # Очистка
    answer = clean_answer(answer)

    return {"answer": answer}
