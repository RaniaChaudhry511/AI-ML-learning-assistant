# AI-ML-learning-assistant

A domain-specific chatbot that helps users learn Artificial Intelligence and Machine Learning. It answers questions about AI, Machine Learning, Deep Learning, NLP, Generative AI, Transformers, and basic Python for AI/ML.

The project shows how a pre-trained Transformer language model can be placed inside an interactive chatbot and guided toward one learning domain using system instructions.

This project is a Google Colab notebook. It is not deployed as a hosted app.

Features

Simple, beginner-friendly explanations with examples
Step-by-step breakdown of difficult concepts
Conceptual comparisons (e.g. CNN vs RNN)
Basic Python code when requested
Conversation history for a more natural learning experience
Gradio chat interface with ready-made example questions
Tech Stack
Language: Python
Model: Qwen/Qwen2.5-0.5B-Instruct
Libraries: PyTorch, Hugging Face Transformers, Gradio
Platform: Google Colab

How It Works

The tokenizer and model are loaded from the Hugging Face Hub.
A system prompt (AI/ML tutor), the chat history and the new message are combined using the model's chat template.
The text is converted into token IDs.
The model predicts the next tokens (max_new_tokens=300, temperature=0.7, top_p=0.9, repetition_penalty=1.1).
Only the new tokens are decoded back into text and shown in the chat.

Installation
bash
pip install torch transformers gradio accelerate
Usage
Open the notebook in Google Colab and set the runtime to GPU (e.g. T4).
Run all cells.
Open the Gradio link printed in the output (public share links last about one week).

Example Questions

Explain NLP in simple words.
What is supervised learning?
Explain how a Transformer works.
What is the difference between CNN and RNN?
What is overfitting and how can we prevent it?
Explain Generative AI with an example.

Limitations

The model is small (0.5B parameters), so answers can be incomplete or wrong.
It has no internet access, so it cannot look up recent information.
The topic focus comes from the system prompt only, so off-topic questions may still get answers.
Verify important concepts with textbooks or official documentation.
Future Improvements
Add RAG with ML books and notes for more accurate answers
Fine-tune on AI/ML Q&A data (LoRA/QLoRA)
Deploy as a permanent app (e.g. Hugging Face Spaces)

Author

Rania Chaudhry
