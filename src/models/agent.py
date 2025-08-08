from typing import Optional

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document


def get_simple_chat_agent(vectorstore, model_name: str = "gemma3:1b"):

    llm = ChatOllama(model=model_name, temperature=0.4)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5, "fetch_k": 20}, search_type="mmr")

    system_prompt = """You are a helpful assistant that answers questions based on the provided document context.

Instructions:
- Use ONLY the retrieved context to answer. If missing, say you don't know.
- Cite quotes or page numbers from the context to support key points.
- Provide a concise, structured answer.

Retrieved Context:
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
                page = d.metadata.get("page")
                prefix = f"[Doc {idx+1}{f', p.{page+1}' if isinstance(page, int) else ''}]"
                return f"{prefix}\n{d.page_content.strip()}"

            context = "\n\n---\n\n".join(format_doc(d, i) for i, d in enumerate(docs))
            response = chain.invoke({
                "context": context,
                "question": query,
                "history": history_messages,
            })
            return response.content
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"

    return chat_with_documents