"""Gradio Blocks layout for the Real World RAG System dashboard.

This module only builds and wires the UI. It knows about `components.py`
(how to render a piece of data as HTML) and `callbacks.py` (what to do on
each interaction), but nothing about JSON files or file paths — that
knowledge lives entirely in `data_loader.py`.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import gradio as gr

import callbacks as cb
import components as c
import data_loader as dl

CSS_PATH = Path(__file__).parent / "styles.css"
CSS = CSS_PATH.read_text(encoding="utf-8")

PIPELINE_VERSION = "v1.0.0"


def build_app() -> gr.Blocks:
    domain_names = dl.get_domain_names()
    default_domain = domain_names[0] if domain_names else None
    default_questions = dl.load_questions_for_domain(default_domain) if default_domain else []
    total_questions = dl.total_question_count()

    with gr.Blocks(title="Real World RAG System") as demo:
        # ---------------- Header ----------------
        gr.HTML(
            """
            <div class="app-header">
              <div class="app-title">Real World RAG System</div>
              <div class="app-subtitle">Interactive Retrieval-Augmented Generation Dashboard</div>
            </div>
            """
        )
        header_badges = gr.HTML(
            c.header_badges_html(total_questions, default_domain or "—", "—", "—")
        )

        # ---------------- State ----------------
        result_state = gr.State(None)

        with gr.Row(equal_height=False):
            # ---------------- Left sidebar (25%) ----------------
            with gr.Column(scale=1, min_width=280):
                with gr.Group(elem_classes=["panel-card"]):
                    gr.Markdown("### 🔍 Search")
                    domain_dropdown = gr.Dropdown(
                        choices=domain_names,
                        value=default_domain,
                        label="Domain",
                    )
                    question_dropdown = gr.Dropdown(
                        choices=default_questions,
                        value=default_questions[0] if default_questions else None,
                        label="Question",
                    )
                    go_button = gr.Button("🚀 Go", variant="primary", size="lg")

                with gr.Accordion("⚙️ Pipeline Configuration", open=False):
                    pipeline_config_html = gr.HTML(c.pipeline_config_html(None))

                with gr.Accordion("🧾 JSON Viewer", open=False):
                    json_viewer = gr.JSON(value=None)

                with gr.Row():
                    download_btn = gr.DownloadButton("⬇️ Download JSON", variant="secondary")
                    export_btn = gr.DownloadButton("📤 Export Results", variant="secondary")

            # ---------------- Right panel (75%) ----------------
            with gr.Column(scale=3):
                question_card = gr.HTML(c.question_card_html(None))

                with gr.Group(elem_classes=["panel-card"]):
                    gr.Markdown("### 📝 Generated Answer")
                    answer_box = gr.Textbox(
                        value="",
                        lines=6,
                        max_lines=14,
                        buttons=["copy"],
                        interactive=False,
                        container=False,
                        placeholder="The generated answer will appear here after you click Go.",
                    )

                gr.Markdown("### 📊 Evaluation Metrics")
                metrics_html = gr.HTML(c.metrics_grid_html(None))

                gr.Markdown("### 📁 Retrieved Documents (Top 4)")
                doc_meta_boxes = []
                doc_code_boxes = []
                for i in range(1, 5):
                    with gr.Accordion(f"📄 Document {i}", open=(i == 1)):
                        meta_html = gr.HTML("<div class='empty-state'>No document loaded.</div>")
                        code_box = gr.Code(
                            value="", language=None, label="Retrieved Text", lines=6, buttons=["copy"]
                        )
                    doc_meta_boxes.append(meta_html)
                    doc_code_boxes.append(code_box)

                ground_truth_comp = gr.HTML(c.ground_truth_html(None))

                gr.Markdown("### 🧩 Pipeline Summary")
                pipeline_summary_comp = gr.HTML(c.pipeline_summary_html(None))

                gr.Markdown("### ⚡ Performance")
                performance_comp = gr.HTML(c.performance_html(None))

        # ---------------- Footer ----------------
        gr.HTML(
            f"""
            <div class="app-footer">
              Dataset: RAGBench &nbsp;•&nbsp; Pipeline Version: {PIPELINE_VERSION}
              &nbsp;•&nbsp; {date.today().isoformat()}
            </div>
            """
        )

        # ---------------- Wiring ----------------
        domain_change_outputs = [
            question_dropdown,
            header_badges,
            question_card,
            answer_box,
            metrics_html,
        ]
        for meta_box, code_box in zip(doc_meta_boxes, doc_code_boxes):
            domain_change_outputs.extend([meta_box, code_box])
        domain_change_outputs.extend(
            [
                ground_truth_comp,
                pipeline_summary_comp,
                performance_comp,
                pipeline_config_html,
                result_state,
                json_viewer,
            ]
        )

        domain_dropdown.change(
            fn=cb.on_domain_change,
            inputs=[domain_dropdown],
            outputs=domain_change_outputs,
        )

        go_click_outputs = [question_card, answer_box, metrics_html]
        for meta_box, code_box in zip(doc_meta_boxes, doc_code_boxes):
            go_click_outputs.extend([meta_box, code_box])
        go_click_outputs.extend(
            [
                ground_truth_comp,
                pipeline_summary_comp,
                performance_comp,
                pipeline_config_html,
                header_badges,
                result_state,
                json_viewer,
            ]
        )

        go_button.click(
            fn=cb.on_go_click,
            inputs=[domain_dropdown, question_dropdown],
            outputs=go_click_outputs,
        )

        download_btn.click(fn=cb.make_download_file, inputs=[result_state], outputs=[download_btn])
        export_btn.click(fn=cb.make_export_csv, inputs=[result_state], outputs=[export_btn])

    return demo
