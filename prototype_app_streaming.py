import gradio as gr
from pathlib import Path
from dotenv import load_dotenv
from implementation.query_data_streaming import stream_answer_question

load_dotenv(override=True)


def format_context(context):
    result = "<h2 style='color: #ff7800;'>Sources</h2>\n\n"
    for doc in context:
        fullSource = Path(doc.metadata["source"])
        lastTwoSource = Path(*fullSource.parts[-2:])
        result += f"<span style='color: #ff7800;'>Source: {lastTwoSource}</span>\n\n"
    return result


def chat(history):
    raw = history[-1]["content"]
    if isinstance(raw, list):
        last_message = " ".join(
            item.get("text", "") for item in raw if isinstance(item, dict)
        )
    else:
        last_message = raw
    prior = history[:-1]

    history.append({"role": "assistant", "content": ""})
    for partial, docs in stream_answer_question(last_message, prior):
        history[-1]["content"] = partial
        yield history, format_context(docs)


def main():
    def put_message_in_chatbot(message, history):
        return "", history + [{"role": "user", "content": message}]

    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title="University Subject Assistant") as ui:
        gr.Markdown("# Ask me anything about subjects and classes")

        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="💬 Conversation",
                    height=600,
                )
                message = gr.Textbox(
                    placeholder="Ask anything about your courses...",
                    show_label=False,
                )

            with gr.Column(scale=1):
                context_markdown = gr.Markdown(
                    label="Retrieved PDFs and Slides",
                    value="*Retrieved context will appear here*",
                    container=True,
                    height=600,
                )

        message.submit(
            put_message_in_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]
        ).then(chat, inputs=chatbot, outputs=[chatbot, context_markdown])

    # ✅ theme moved here, from Blocks() to launch()
    ui.launch(inbrowser=True, share=False, theme=theme)


if __name__ == "__main__":
    main()

print('finished')