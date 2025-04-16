
# Template for LangGraph Integration with Gradio

This repository provides a simple template demonstrating how to integrate a LangGraph application with a Gradio front end. The implementation uses placeholder tools to showcase the wiring between the components.

### What this project demonstrates:
1. How to run a LangGraph graph using [Ollama](https://ollama.com/)
2. How to integrate basic placeholder tools into a LangGraph workflow
3. How to connect the LangGraph application to a Gradio UI and display chat-based output

---

## Prerequisites

Before running this project, ensure you have the following installed:

- **[Python 3.12+](https://www.python.org/downloads/)**  
  This project requires Python 3.12 or higher.

- **[Ollama](https://ollama.com/)**  
  Ollama is used to run the language model locally.

- **`qwen2:7b` model**  
  After installing Ollama, pull the model with:
  ```bash
  ollama pull qwen2:7b
  ```

- **[Poetry](https://python-poetry.org/docs/#installation)**  
  Poetry is used for dependency management and virtual environments.

---

## Installation Instructions

1. **Install Python 3.10+** (if not already installed)  
   You can check your Python version with:
   ```bash
   python3 --version
   ```

2. **Install Ollama**  
   Follow the instructions on the [Ollama website](https://ollama.com/download) to install for your platform.

3. **Pull the model**  
   Once Ollama is installed, pull the `qwen2:7b` model:
   ```bash
   ollama pull qwen2:7b
   ```

4. **Install Poetry**  
   If you don’t have Poetry installed:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

5. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/langgraph-gradio-template.git
   cd langgraph-gradio-template
   ```

6. **Install dependencies with Poetry**
   ```bash
   poetry install
   ```

7. **Run the application**
   ```bash
   poetry run python app.py
   ```

---

## Basic Gradio Integration

Gradio expects a function that accepts a `(user_message, chat_history)` and returns an updated `chat_history` and an empty input string. In this template, we use the LangGraph app inside a `chat_fn` to process user input and return results:

```python
def chat_fn(message, history):
    session_id = "session-123"

    result = graph_app.invoke(
        {"messages": [HumanMessage(content=message)]},
        config={"configurable": {"thread_id": session_id}}
    )

    response = result["messages"][-1].content
    history.append((message, response))
    return history, ""
```

We then define a simple Gradio interface using [Blocks](https://www.gradio.app/docs/gradio/blocks):

```python
with gr.Blocks() as demo:
    gr.Markdown("### LangGraph Chat with Gradio")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Type your message")
    send_btn = gr.Button("Send")

    send_btn.click(fn=chat_fn, inputs=[msg, chatbot], outputs=[chatbot, msg])
```

This creates a basic UI with:
- A `Chatbot` display area
- A `Textbox` for input
- A `Send` button to trigger message handling

> Note: The tools in this template are placeholders. For example, you can ask it about the weather, but the response is hard-coded rather than coming from a real weather API.

