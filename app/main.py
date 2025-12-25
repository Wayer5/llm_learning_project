from fastapi import FastAPI
from pydantic import BaseModel

from .llm import ask_llm
from .rag import retrieve_context

app = FastAPI()


class Question(BaseModel):
    question: str


def clean_answer(text: str) -> str:
    """
    Обрезает лишние фразы и повторения, оставляя короткий связный ответ.
    """
    # Можно дополнить правилами очистки по твоей логике
    for sep in [",", "но", "\n"]:
        if sep in text:
            text = text.split(sep)[0]
            break
    return text.strip().capitalize()


@app.get("/")
def root():
    return {"status": "RAG + LLM локально работает"}


@app.post("/ask")
def ask(question: Question):
    context = retrieve_context(question.question)
    # Собираем prompt для модели
    prompt_text = (
        "Используй этот контекст для ответа на вопрос:\n"
        + "\n".join(context)
        + f"\n\nВопрос: {question.question}\nОтвет:"
    )

    # messages для ask_llm теперь одна запись с пользователем
    messages = [{"role": "user", "content": prompt_text}]

    answer = ask_llm(messages, max_length=150)
    answer = clean_answer(answer)
    return {"answer": answer}
