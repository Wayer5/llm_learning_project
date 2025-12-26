import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

MODEL_NAME = "tinyllama/TinyLlama-1.1B-Chat-v1.0"

# Загружаем модель и токенизатор
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Создаём pipeline генерации
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    torch_dtype=torch.float16
)

# Инструкция для модели
INSTRUCTION = (
    "Ты эксперт по IT-технологиям. Отвечай строго на русском.\n"
    "Используй информацию из контекста для ответа.\n"
    "Если ответа нет в контексте — скажи 'Нет информации в контексте'.\n"
    "Не используй слово 'Контекст:' в ответе.\n\n"
)


def ask_llm(messages, max_length=150):
    """
    Генерация ответа модели на русском языке на основе сообщений.
    messages: [{"role": "user"/"system", "content": "..."}]
    """
    if not generator:
        raise RuntimeError("Модель не загружена")

    prompt = "\n".join([m["content"] for m in messages])
    output = generator(
        prompt,
        max_new_tokens=max_length,
        do_sample=False,       # детерминированно
        num_beams=1,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.5
    )

    text = output[0]["generated_text"].strip()

    # Вырезаем инструкцию, если модель вставила её в ответ
    if text.startswith(INSTRUCTION.strip()):
        text = text[len(INSTRUCTION):].strip()

    # Берём только первую непустую строку
    lines = [line for line in text.splitlines() if line.strip()]
    if lines:
        text = lines[0].strip()

    return text.capitalize()
