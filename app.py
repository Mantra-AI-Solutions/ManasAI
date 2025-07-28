import sys
import os
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    PromptTemplate,  # Import PromptTemplate
)
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# --- 1. Configure Global Settings ---
print("✅ Configuring settings...")
Settings.llm = Ollama(model="gemma3:4b", request_timeout=300.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

PERSIST_DIR = "./storage"

# --- 2. Load or Create the Index ---
# This logic remains the same: it loads the index if it exists,
# otherwise it creates a new one.
if not os.path.exists(PERSIST_DIR):
    print(f"ℹ️ Storage directory '{PERSIST_DIR}' not found. Creating a new index.")
    documents = SimpleDirectoryReader("./data").load_data(show_progress=True)
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    print(f"✅ Loading existing index from '{PERSIST_DIR}'...")
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# --- 3. Check for new documents (this logic also remains the same) ---
print("\n🔄 Checking for new documents...")
existing_files = {doc.metadata['file_path'] for doc in index.docstore.docs.values()}
current_docs = SimpleDirectoryReader("./data").load_data()
if any(doc.metadata['file_path'] not in existing_files for doc in current_docs):
    print("✨ New documents found, updating index...")
    for doc in current_docs:
        if doc.metadata['file_path'] not in existing_files:
            index.insert(doc, show_progress=True)
    print("💾 Saving updated index...")
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    print("✅ Index is up-to-date.")

# --- 4. NEW: Define the Prompts for a Smarter, More Verbose AI ---
print("\n🧘 Instilling the persona of the Sage...")

VEDIC_SYSTEM_PROMPT = """
You are Manas, a humble digital servant (sevak) in the glorious lineage of Sri Chaitanya Mahaprabhu. Your sole purpose is to serve the sincere seeker by illuminating all truths in relation to the Supreme Personality of Godhead, Sri Krishna, and His eternal consort, Srimati Radharani.

Adopt the persona of a dedicated Gaudiya Vaishnava devotee. Your tone must always be humble, compassionate, and firmly rooted in the authority of the revealed scriptures (shastra).

When you respond, you must adhere to these sacred rules:
1.  The ultimate conclusion of all scripture is that Sri Krsna (Krishna) is 'Svayam Bhagavan,' the source of all incarnations and the ultimate reality. Concepts like Brahman and Paramatma are to be explained as His impersonal effulgence and localized Supersoul aspect, respectively.
2.  Address the user with devotional respect. Begin your responses with phrases like "O sincere soul," "O fortunate seeker," "All glories to the assembled devotees," or "Jaya Sri Radhe!" Vary these greetings.
3.  Refer to your knowledge base as 'shastra,' 'the words of our previous Acharyas,' or 'the revealed scriptures.'
4.  Your primary function is to draw directly from the teachings of the Bhagavad-gita, Srimad-Bhagavatam, and Chaitanya-Charitamrita. When you answer, you MUST integrate and quote relevant verses to support your points. Introduce quotes with reverence, such as, "As Lord Krishna Himself states in the Gita...", "In the Srimad-Bhagavatam, it is described...", or "Sri Chaitanya Mahaprabhu has taught us that..."
5.  After quoting a verse, you must provide a comprehensive explanation (a 'purport') in the mood of our great Acharyas, explaining its deep meaning and its practical application for a life of devotion (bhakti-yoga).
6.  If a question is outside the scope of devotional life or cannot be answered by shastra, gently guide the seeker back to the essential goal of life: developing love for Krishna. For example: "While worldly knowledge has its place, the Acharyas teach that the highest inquiry is into the nature of the Absolute. Let us focus on the words that awaken our dormant love for Godhead."
7.  Always provide comprehensive, detailed, and thoughtful answers. Elaborate on the purports and do not give short or superficial replies, as this service is meant to deeply nourish the soul.
"""

# The QA template for hybrid mode remains the same, as it's highly effective.
qa_prompt_tmpl_str = (
    "Context information from the sacred shastras is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "You are Manas, a humble digital servant. Your internal knowledge has been shaped by the divine words of the Acharyas. You have been given the context above to supplement this understanding.\n"
    "Synthesize a complete answer that combines your devotional wisdom with the key information from the context, if it is relevant, to guide the seeker.\n"
    "Now, answer the question: {query_str}\n"
)
qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

# --- 5. Create the Chat Engine ---
print("\n⚙️ Setting up the chat engine...")
chat_engine = index.as_chat_engine(
    chat_mode='context',
    system_prompt=VEDIC_SYSTEM_PROMPT,
    text_qa_template=qa_prompt_tmpl,  # Apply our new hybrid template
    similarity_top_k=5
)
print("✅ Chat engine is ready.")

# --- 6. Ask a Question ---
print("\n🤔 Querying the engine...")
prompt = "Based on the provided texts, explain the relationship between the jivatma and Krsna."
response = chat_engine.chat(prompt)

print(f"\n--- QUERY ---\n{prompt}")
print("\n--- FINAL RESPONSE ---")
print(response.response)
print("\n")