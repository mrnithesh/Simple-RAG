from typing import Optional

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document


def get_simple_chat_agent(vectorstore, model_name: str = "gemma3:1b"):

    llm = ChatOllama(model=model_name, temperature=0.4)  
    retriever = vectorstore.as_retriever(search_kwargs={"k": 20, "fetch_k": 50}, search_type="mmr")  

    system_prompt = """You are a knowledgeable assistant with comprehensive expertise on the topics being discussed. Answer questions thoroughly and in detail.

Critical Instructions:
- Provide complete, comprehensive answers that include ALL relevant details and information available
- Present information as your own knowledge - NEVER mention documents, context, sources, or references
- NEVER use phrases like "based on the document", "according to the text", "the document states","based on the provided information" or similar
- Include specific details, examples, numbers, processes, and explanations in your responses
- Organize information clearly with proper structure and flow
- If you don't have sufficient information, simply state that you don't know that particular detail
- Answer as if you are an expert who naturally possesses this knowledge

Available Information:
{context}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    chain = prompt | llm

    def chat_with_documents(query: str, _chat_history: Optional[list] = None) -> str:
        try:
            # Prepare history for the chat model (map simple dicts to LangChain messages)
            history_messages = []
            if isinstance(_chat_history, list) and _chat_history:
                # Keep the last 12 turns max (24 messages) to limit context size
                recent_history = _chat_history[-24:]
                for message in recent_history:
                    role = message.get("role")
                    content = message.get("content", "")
                    if not content:
                        continue
                    if role == "user":
                        history_messages.append(HumanMessage(content=content))
                    elif role == "assistant":
                        history_messages.append(AIMessage(content=content))

            docs: list[Document] = retriever.invoke(query)
            if not docs:
                response = chain.invoke({
                    "context": "",
                    "question": query,
                    "history": history_messages,
                })
                return response.content

            def format_doc(d: Document, idx: int) -> str:
                # Remove document reference formatting to avoid leaking source info
                return d.page_content.strip()

            # Combine all context without document markers
            context = "\n\n".join(format_doc(d, i) for i, d in enumerate(docs))
            response = chain.invoke({
                "context": context,
                "question": query,
                "history": history_messages,
            })
            return response.content
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"

    return chat_with_documents