"""Entry point for the Real World RAG System dashboard.

    python gradio_app/app.py
"""

from __future__ import annotations

import gradio as gr

from ui import CSS, build_app

if __name__ == "__main__":
    demo = build_app()
    demo.launch(css=CSS, theme=gr.themes.Soft())
