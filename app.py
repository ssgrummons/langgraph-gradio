from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langchain_ollama.chat_models import ChatOllama


from tools import search_tool, weather_info_tool, hub_stats_tool
import gradio as gr
from smolagents import GradioUI, LiteLLMModel
from retriever import load_guest_dataset
import yaml
from dotenv import load_dotenv

# Import our custom tools from their modules
from retriever import load_guest_dataset

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)

chat = ChatOllama(model="qwen2:7b", verbose=True)
tools = [
        load_guest_dataset, 
        search_tool, 
        weather_info_tool, 
        hub_stats_tool
        ]
chat_with_tools = chat.bind_tools(tools)

# Generate the AgentState and Agent graph
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def assistant(state: AgentState):
    return {
        "messages": [chat_with_tools.invoke(state["messages"])],
    }

## The graph
builder = StateGraph(AgentState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message requires a tool, route to tools
    # Otherwise, provide a direct response
    tools_condition,
)
builder.add_edge("tools", "assistant")
graph_app = builder.compile()

graph_state = {}

# Gradio expects a function with (chat_history, user_message) -> (updated_chat_history)
def chat_fn(message, history):
    session_id = "session-123"

    result = graph_app.invoke(
        {"messages": [HumanMessage(content=message)]},
        config={"configurable": {"thread_id": session_id}}
    )

    response = result["messages"][-1].content
    history.append((message, response))
    return history, ""


# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("### LangGraph Chat with Gradio")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Type your message")
    send_btn = gr.Button("Send")

    # Hook the send button
    send_btn.click(fn=chat_fn, inputs=[msg, chatbot], outputs=[chatbot, msg])

if __name__ == "__main__":
    demo.launch()