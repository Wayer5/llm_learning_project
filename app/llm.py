import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "ai-forever/mGPT"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)


def ask_llm(messages, max_length=100):
    """
    messages: [{"role": "user"/"system", "content": "..."}]
    Возвращает один связный ответ модели без повторов.
    """
    # Формируем prompt: только контекст + вопрос (без "Ответь...")
    prompt = "\n".join([m["content"] for m in messages])

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
        repetition_penalty=2.0
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Убираем префикс "Ответ:" если есть
    if "Ответ:" in text:
        text = text.split("Ответ:", 1)[-1]
    return text.strip()
