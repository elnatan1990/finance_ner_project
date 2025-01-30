import gradio as gr
from rag_system.rag_agent import create_rag_system
from ner.ner import NERPredictor

# Initialize the RAG system and NER model
qa = create_rag_system()
ner = NERPredictor.load_model("../ner/ner_model")


def process_text(text):
    # Get RAG response
    rag_response = qa.run(text)

    # Get NER predictions
    predictions = ner.predict(rag_response)

    # Define colors for each entity type
    colors = {
        "company": "#FF6B6B",  # Red
        "name": "#4CAF50",  # Green
        "street_address": "#FF9800"  # Orange
    }

    # Sort predictions by length (longest first) to handle overlapping entities
    predictions.sort(key=lambda x: len(x[0]), reverse=True)

    # Create a list of all character positions and their corresponding HTML tags
    markers = []
    entity_list = []

    for entity_text, label in predictions:
        start_idx = 0
        while True:
            # Find next occurrence of entity
            pos = rag_response.find(entity_text, start_idx)
            if pos == -1:  # No more occurrences
                break

            # Add opening and closing tags to markers
            markers.append((pos,
                            f"<span style='background-color: {colors[label]}; padding: 2px 5px; border-radius: 3px; color: white;'>"))
            markers.append((pos + len(entity_text), "</span>"))

            # Move start_idx for next iteration
            start_idx = pos + 1

        # Add to entity list
        entity_list.append(
            f"<div style='margin: 5px 0;'><span style='background-color: {colors[label]}; "
            f"padding: 2px 5px; border-radius: 3px; color: white;'>{label.upper()}</span>: {entity_text}</div>"
        )

    # Sort markers by position
    markers.sort(key=lambda x: (x[0], -len(x[1])))  # Sort by position, longer tags first

    # Build the highlighted text
    result = []
    last_pos = 0

    for pos, tag in markers:
        if pos > last_pos:
            result.append(rag_response[last_pos:pos])
        result.append(tag)
        last_pos = pos

    # Add any remaining text
    if last_pos < len(rag_response):
        result.append(rag_response[last_pos:])

    colored_text = ''.join(result)

    # Create entity section if entities were found
    entity_section = (
            "<div style='margin-top: 20px; padding: 10px; background-color: #f5f5f5; border-radius: 5px;'>"
            "<h3>Entities Found:</h3>" + "".join(entity_list) + "</div>"
    ) if entity_list else ""

    # Combine everything into final HTML
    final_html = (
        f"<div style='font-family: Arial, sans-serif;'>"
        f"<div style='margin-bottom: 20px; line-height: 1.5;'>{colored_text}</div>"
        f"{entity_section}</div>"
    )

    return final_html


# Create Gradio interface
iface = gr.Interface(
    fn=process_text,
    inputs=gr.Textbox(
        label="Ask a question about finance or business",
        lines=3,
        placeholder="Enter your question here..."
    ),
    outputs=gr.HTML(label="Response with Named Entities"),
    title="Finance Q&A with Named Entity Recognition",
    description="Ask questions and see named entities (companies, names, addresses) highlighted in the response.",
    examples=[
        ["explain **GreenTech Inc. Sustainability Financial Statement**"],
        ["what info you know about Dear Lorraine Fischer-Maurice"],
        ["what info you know about Title: Network Outage - Angelika Stroh"],
        ["Tell me about Microsoft's CEO Satya Nadella."],
        ["What companies did John Smith work for?"],
        ["When Cameron-Mcknight Supply Chain Management Agreement entered into place?"],
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    iface.launch()