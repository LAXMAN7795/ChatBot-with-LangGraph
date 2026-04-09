import os
import gradio as gr

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_groq import ChatGroq

# ✅ Load API key
groq_api_key = os.getenv("GROQ_API_KEY")

# ✅ Initialize LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)

# ✅ State
class State(TypedDict):
    messages: Annotated[list, add_messages]

# ✅ Graph
graph_builder = StateGraph(State)

def chatbot(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": response}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

# ✅ Chat function
def chat_with_bot(user_input, history):
    history = history or []

    messages = []
    for human, ai in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": ai})

    messages.append({"role": "user", "content": user_input})

    result = graph.invoke({"messages": messages})

    response = result["messages"][-1].content

    history.append((user_input, response))

    return history, history

# ✅ UI
with gr.Blocks() as demo:
    gr.Markdown("## 🤖 LangGraph Chatbot (Groq Powered)")
    
    chatbot_ui = gr.Chatbot()
    msg = gr.Textbox(placeholder="Type your message...")
    clear = gr.Button("Clear")

    msg.submit(chat_with_bot, [msg, chatbot_ui], [chatbot_ui, chatbot_ui])
    clear.click(lambda: None, None, chatbot_ui, queue=False)

# ✅ IMPORTANT for Render
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
