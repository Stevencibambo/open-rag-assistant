from chroma_url_rag import ConversationalChromaRAG

# Initialize conversational RAG
chat_rag = ConversationalChromaRAG(
    model_name="llama3.2",
    chroma_persist_dir="./chat_db"
)

# Load some websites
chat_rag.load_multiple_websites([
    "https://www.kadea.academy/formations/dev-web/?city=kinshasa",
    "https://www.kadea.academy/formations/dev-ia/?city=kinshasa",
    'https://www.kadea.academy/formations/marketing-digital/?city=kinshasa'
])

# Have a conversation
while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ['quit', 'exit', 'bye']:
        break
    
    result = chat_rag.chat_with_context(user_input)
    
    print(f"\nAssistant: {result['response']}")
    print(f"\n[Used {len(result['sources'])} sources]")