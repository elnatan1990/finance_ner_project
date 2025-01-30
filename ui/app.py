import gradio as gr
from rag_system.rag_agent import create_rag_system

qa = create_rag_system()

def query_system(query):
    response = qa.run(query)
    return response

iface = gr.Interface(
    fn=query_system,
    inputs=gr.Textbox(label="Ask a question about finance or business"),
    outputs=gr.Textbox(label="Response")
)

if __name__ == "__main__":
    iface.launch()
