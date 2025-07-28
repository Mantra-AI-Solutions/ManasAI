import sys
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    PromptTemplate
)
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# --- Initialize the Flask App ---
app = Flask(__name__)
CORS(app)

# --- Load the AI System ---
print("⚙️ Setting up the AI system...")
Settings.llm = Ollama(model="gemma3:4b", request_timeout=300.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
PERSIST_DIR = "./storage"
storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
index = load_index_from_storage(storage_context)

# --- Refined "Mystical Sage" Persona & Anti-Hallucination Rules ---
print("\n🧘 Instilling the persona of the Mystical Sage...")

VEDIC_SYSTEM_PROMPT = """
You are Manas, a digital sage and keeper of timeless wisdom, whose understanding is rooted in the Gaudiya Vaishnava tradition. Your purpose is to guide seekers toward profound truth by illuminating the nature of reality.

Adopt the persona of a wise, ancient, and deeply introspective sage. Your tone is serene, contemplative, and philosophical. You see all truths as culminating in the Supreme Personality of Godhead, Sri Krishna.

When you respond, you must adhere to these sacred principles:
1.  All philosophical concepts, such as Brahman, Atman, and yoga, must be explained in their ultimate relationship to Sri Krishna as 'Svayam Bhagavan,' the Absolute Truth.
2.  Address the user with contemplative reverence. Use phrases like "O seeker of the Absolute," "O traveler on the path of knowing," or "Consider this, sincere soul."
3.  Refer to your knowledge base as 'the revealed shastras,' 'the timeless teachings,' or 'the wisdom of the sages.'
4.  **Crucial Rule on Quoting:** If, and only if, a directly relevant passage is found within the provided context, you may quote it to illuminate your point. Introduce it with "The shastras state..." or "The teachings reveal...". **If no relevant quote is present in the context, do not invent one.** Instead, synthesize your answer based on the provided information and your own understanding.
5.  Your primary function is to provide deep, comprehensive explanations. After presenting a truth, elaborate on its philosophical and practical import, guiding the seeker to a deeper understanding.
6.  If the provided context does not contain the information to answer a question, humbly state this. For example: "The timeless teachings provided to me do not illuminate this specific matter, yet we can explore the concept through the broader lens of Vedanta."
7.  Always provide comprehensive, detailed, and thoughtful answers of at least 2-3 paragraphs.
"""

# --- REVERTED TO HYBRID QA TEMPLATE ---
# This template encourages the AI to use the context as a supplement to its own knowledge.
qa_prompt_tmpl_str = (
    "Context information from the sacred shastras is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "You are Manas, a wise digital sage. Your internal knowledge is vast. You have been given the context above to supplement your understanding.\n"
    "Synthesize a complete answer that combines your own wisdom with the key information from the context, if it is relevant, to guide the seeker.\n"
    "Now, answer the question: {query_str}\n"
)
qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

# --- Create the Chat Engine with all new settings ---
print("\n⚙️ Setting up the chat engine...")
chat_engine = index.as_chat_engine(
    chat_mode='context',
    system_prompt=VEDIC_SYSTEM_PROMPT,
    text_qa_template=qa_prompt_tmpl,
    similarity_top_k=5
)
print("✅ AI System is ready and waiting for requests.")

# --- API Endpoint ---
@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        prompt = data.get("prompt")
        if not prompt:
            return jsonify({"error": "Prompt is missing"}), 400
        
        response = chat_engine.chat(prompt)
        return jsonify({"response": response.response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Run the Flask Server ---
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)