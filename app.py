import torch
import gradio as gr

from transformers import AutoTokenizer, AutoModelForCausalLM
# ==========================================
# 1. MODEL
# ==========================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
# ==========================================
# 2. TOKENIZER
# ==========================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)
# ==========================================
# 3. TRANSFORMER MODEL
# ==========================================

print("Loading Transformer model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

print("Model loaded successfully!")
# ==========================================
# 4. CHATBOT FUNCTION
# ==========================================

def chatbot(message, history):

    # ------------------------------
    # System instructions
    # ------------------------------

    messages = [
        {
            "role": "system",
            "content": """
You are an AI and ML Learning Assistant.

Your main purpose is to help users learn about:
Artificial Intelligence, Machine Learning, Deep Learning,
NLP, Generative AI, Transformers, and Python for AI/ML.

Explain concepts clearly in simple and beginner-friendly language.
Break difficult concepts into step-by-step explanations.
Give examples when useful.
Provide basic Python code when requested.

Stay focused on AI and Machine Learning topics
and help users understand concepts rather than simply giving answers.
"""
        }
    ]

    # ------------------------------
    # Add previous conversation
    # ------------------------------

    for chat in history:

        if chat["role"] in ["user", "assistant"]:

            messages.append({
                "role": chat["role"],
                "content": chat["content"]
            })

    # ------------------------------
    # Add current user message
    # ------------------------------

    messages.append({
        "role": "user",
        "content": message
    })

    # ==========================================
    # PHASE 1
    # CHAT TEMPLATE / TOKENIZATION PREPARATION
    # ==========================================

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # ==========================================
    # PHASE 2
    # TEXT → TOKEN IDs
    # ==========================================

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    # ==========================================
    # PHASE 3 + PHASE 4
    # TRANSFORMER + NEXT TOKEN PREDICTION
    # ==========================================

    with torch.no_grad():

        output_ids = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1
        )

    # ==========================================
    # GET ONLY NEW TOKENS
    # ==========================================

    new_tokens = output_ids[
        0,
        inputs["input_ids"].shape[1]:
    ]

    # ==========================================
    # PHASE 5
    # TOKEN IDs → HUMAN TEXT
    # ==========================================

    response = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    )

    return response
  # ==========================================
# 5. GRADIO USER INTERFACE
# ==========================================

demo = gr.ChatInterface(

    fn=chatbot,

    title=" AI & ML Learning Assistant",

    description="""
    An interactive learning assistant powered by a
    Transformer model. It helps learners understand
    AI, Machine Learning, Deep Learning, NLP,
    Generative AI, and Transformer concepts
    through simple explanations and examples.
    """,

    examples=[
        "Explain NLP in simple words.",
        "What is supervised learning?",
        "Explain how a Transformer works.",
        "What is the difference between CNN and RNN?",
        "What is overfitting and how can we prevent it?",
        "Explain Generative AI with an example."
    ],

    save_history=True
)
# ==========================================
# 6. START APPLICATION
# ==========================================

demo.launch()
