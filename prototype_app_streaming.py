import gradio as gr
import gradio as gr
from pathlib import Path

from dotenv import load_dotenv

from implementation.query_data_streaming import stream_answer_question

load_dotenv(override=True)


def format_context(context):
    result = "<h2 style='color: #ff7800;'>Soruces</h2>\n\n"
    for doc in context:
        fullSource = Path(doc.metadata["source"])
        lastTwoSource = Path(*fullSource.parts[-2:])


        result += f"<span style='color: #ff7800;'>Source: {lastTwoSource}</span>\n\n"

        # result += doc.page_content + "\n\n"
    return result


def chat(history):
    last_message = history[-1]["content"]
    prior = history[:-1]
    
    history.append({"role": "assistant", "content":""})
    for partial, docs in stream_answer_question(last_message, prior):
        history[-1]["content"] = partial
        yield history, format_context(docs)


def main():
    def put_message_in_chatbot(message, history):
        return "", history + [{"role": "user", "content": message}]

    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title="Insurellm Expert Assistant", theme=theme) as ui:
        gr.Markdown("# Ask me anything about subjects and classes")

        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="💬 Conversation", height=600, type="messages"
                )
                message = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask anything about Insurellm...",
                    show_label=False,
                )

            with gr.Column(scale=1):
                context_markdown = gr.Markdown(
                    label="Retrieved pdfs and slides",
                    value="*Retrieved context will appear here*",
                    container=True,
                    height=600,
                )
   
        message.submit(
            put_message_in_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]
        ).then(chat, inputs=chatbot, outputs=[chatbot, context_markdown])

    ui.launch(inbrowser=True, share=False)


if __name__ == "__main__":
    main()

print('finishes')