import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from dotenv import load_dotenv

load_dotenv()

import requests
import gradio as gr

from core.inference import predict_disease

# ─────────────────────────────────────────────────────────────────────────────
# GROQ AI CHATBOT
# ─────────────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def chat_with_ai(
    user_message: str,
    plant_name: str = None,
    disease_name: str = None,
    confidence: float = None,
) -> dict:
    """
    Chatbot endpoint for interactive AI conversation about plant disease.
    Returns a dict with keys: success, message, error (if applicable).
    """
    if not GROQ_API_KEY:
        return {"success": False, "error": "Groq API key not configured."}

    # Build context if plant info is provided
    context = ""
    if plant_name and disease_name:
        context = (
            f"Context: User detected a plant as {plant_name} with {disease_name} condition "
            f"(confidence: {confidence:.1f}%).\n\n"
        )

    prompt = (
        f"You are an expert plant pathologist and agricultural advisor. "
        f"Help the user with sustainable farming and organic treatment advice.\n\n"
        f"{context}"
        f"User: {user_message}"
    )

    try:
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 500,
        }

        response = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=20,
        )

        if response.status_code != 200:
            error_detail = response.text if response.text else response.reason
            return {
                "success": False,
                "error": f"API Error ({response.status_code}): {error_detail}",
            }

        result = response.json()
        message = result["choices"][0]["message"]["content"].strip()
        return {"success": True, "message": message}

    except requests.exceptions.Timeout:
        return {"success": False, "error": "API timeout. Try again."}
    except Exception as e:
        return {"success": False, "error": f"Error: {str(e)}"}


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS (loaded from assets/styles.css)
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSS_PATH = os.path.join(BASE_DIR, "assets", "styles.css")

with open(CSS_PATH, "r", encoding="utf-8") as f:
    custom_css = f.read()

# State to hold last detection result for chatbot context
_last_detection = {"plant": None, "disease": None, "confidence": None}


# ─────────────────────────────────────────────────────────────────────────────
# ROUTING HANDLERS
# ─────────────────────────────────────────────────────────────────────────────
def handle_prediction(image):
    """Handle prediction and format outputs as HTML treatment panels."""
    plant_name, disease_name, confidence_str, treatments, error_msg = predict_disease(
        image
    )

    empty_bio = '<div class="treatment-panel bio"><h4>🧫 Biological Treatments</h4><p class="treatment-placeholder">No data.</p></div>'
    empty_org = '<div class="treatment-panel org"><h4>🌾 Organic Treatments</h4><p class="treatment-placeholder">No data.</p></div>'
    empty_cul = '<div class="treatment-panel cul"><h4>🌱 Cultural Practices</h4><p class="treatment-placeholder">No data.</p></div>'

    if error_msg:
        return "", "", "", empty_bio, empty_org, empty_cul, error_msg

    _last_detection["plant"] = plant_name
    _last_detection["disease"] = disease_name
    _last_detection["confidence"] = confidence_str

    def fmt_html(items, panel_class, icon, title):
        items_html = (
            "".join(f"<li>{item}</li>" for item in items)
            if items
            else "<li>No recommendations available.</li>"
        )
        return (
            f'<div class="treatment-panel {panel_class}">'
            f"<h4>{icon} {title}</h4>"
            f'<ul class="treatment-list">{items_html}</ul>'
            f"</div>"
        )

    bio = fmt_html(
        treatments.get("biological", []), "bio", "🧫", "Biological Treatments"
    )
    org = fmt_html(treatments.get("organic", []), "org", "🌾", "Organic Treatments")
    cul = fmt_html(treatments.get("cultural", []), "cul", "🌱", "Cultural Practices")

    return (
        plant_name,
        disease_name,
        confidence_str,
        bio,
        org,
        cul,
        "✅ Analysis complete! Check the results above.",
    )


def handle_chat(user_msg, history):
    """Handle LLM chatbot messages."""
    if not user_msg.strip():
        return history, ""

    plant = _last_detection.get("plant")
    disease = _last_detection.get("disease")
    confidence = _last_detection.get("confidence")

    # Parse confidence float if available
    conf_float = None
    if confidence:
        try:
            conf_float = float(confidence.replace("%", ""))
        except Exception:
            pass

    result = chat_with_ai(user_msg, plant, disease, conf_float)

    if result.get("success"):
        reply = result["message"]
    else:
        reply = f"⚠️ {result.get('error', 'Unknown error')}"

    history = history or []
    history.append((user_msg, reply))
    return history, ""


# ─────────────────────────────────────────────────────────────────────────────
# GRADIO UI WITH CUSTOM STYLING
# ─────────────────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="🌱 Plant Disease Detector",
    css=custom_css,
    theme=gr.themes.Soft(primary_hue="green"),
) as app:

    # ── Header ──────────────────────────────────────────────────────────────
    gr.HTML("""
        <div class="header-section">
            <div style="font-size:3rem; margin-bottom:0.3rem;">🌿</div>
            <h1>Plant Disease Detector</h1>
            <p>Upload a photo of your plant leaf and get instant AI-powered disease diagnosis
               along with sustainable, eco-friendly treatment recommendations.</p>
        </div>
    """)

    # ── Tabs ────────────────────────────────────────────────────────────────
    with gr.Tabs(elem_classes="tabs"):

        # ── TAB 1: Analyze ──────────────────────────────────────────────────
        with gr.Tab("🔬 Analyze Plant"):

            with gr.Row(equal_height=False):

                # Left: upload
                with gr.Column(scale=1, min_width=320):
                    gr.HTML('<p class="section-label">📸 Upload Leaf Image</p>')
                    image_input = gr.Image(
                        type="pil",
                        label="Drop your leaf image here",
                        elem_classes="upload-section",
                        height=360,
                        show_label=False,
                    )
                    predict_button = gr.Button(
                        "🔍 Analyze Plant",
                        variant="primary",
                        size="lg",
                        elem_classes="predict-button",
                    )
                    gr.HTML("""
                        <div style="margin-top:0.8rem; padding:0.8rem 1rem;
                                    background:#f0fdf4; border-radius:10px;
                                    font-size:0.88rem; color:#4a4a4a; line-height:1.8;">
                            <b>✨ Supported formats:</b> JPEG, PNG, WebP<br>
                            <b>📏 Max file size:</b> 16 MB
                        </div>
                    """)

                # Right: detection results
                with gr.Column(scale=1, min_width=320):
                    gr.HTML('<p class="section-label">🎯 Detection Results</p>')

                    with gr.Group(elem_classes="result-card"):
                        plant_output = gr.Textbox(
                            label="🌿 Plant Name",
                            interactive=False,
                            placeholder="Plant type will appear here…",
                        )
                        disease_output = gr.Textbox(
                            label="🦠 Disease / Condition",
                            interactive=False,
                            placeholder="Disease diagnosis will appear here…",
                        )
                        confidence_output = gr.Textbox(
                            label="📊 Confidence Score",
                            interactive=False,
                            placeholder="Confidence level will appear here…",
                        )

            # ── Treatment recommendations ─────────────────────────────────
            gr.HTML(
                '<p class="section-label" style="margin-top:1.4rem;">💊 Sustainable Treatment Recommendations</p>'
            )

            with gr.Row():
                with gr.Column(scale=1):
                    bio_output = gr.HTML(
                        value='<div class="treatment-panel bio"><h4>🧫 Biological Treatments</h4>'
                        '<p class="treatment-placeholder">Run analysis to see recommendations…</p></div>'
                    )

                with gr.Column(scale=1):
                    org_output = gr.HTML(
                        value='<div class="treatment-panel org"><h4>🌾 Organic Treatments</h4>'
                        '<p class="treatment-placeholder">Run analysis to see recommendations…</p></div>'
                    )

                with gr.Column(scale=1):
                    cul_output = gr.HTML(
                        value='<div class="treatment-panel cul"><h4>🌱 Cultural Practices</h4>'
                        '<p class="treatment-placeholder">Run analysis to see recommendations…</p></div>'
                    )

            # Status
            error_output = gr.Textbox(
                label="⚠️ Status",
                interactive=False,
                elem_classes="status-box",
                show_label=True,
            )

            # Wire up button
            predict_button.click(
                fn=handle_prediction,
                inputs=[image_input],
                outputs=[
                    plant_output,
                    disease_output,
                    confidence_output,
                    bio_output,
                    org_output,
                    cul_output,
                    error_output,
                ],
            )

        # ── TAB 2: AI Chat ──────────────────────────────────────────────────
        with gr.Tab("🤖 Ask AI Assistant"):

            gr.HTML("""
                <div class="tip-box">
                    <b>💡 Tip:</b> Analyze a plant first, then ask follow-up questions here.
                    The AI will automatically use the detected plant and disease as context.
                </div>
            """)

            chatbot = gr.Chatbot(
                label="Plant Disease Assistant",
                height=400,
                elem_classes="chatbot-area",
            )

            with gr.Row(elem_classes="chat-input-row"):
                chat_input = gr.Textbox(
                    placeholder="Ask anything about plant diseases, treatments, prevention…",
                    label="Your question",
                    scale=5,
                    show_label=False,
                )
                send_btn = gr.Button(
                    "Send 📤",
                    variant="primary",
                    scale=1,
                    min_width=90,
                    elem_classes="send-btn",
                )

            gr.Examples(
                examples=[
                    "What causes this disease and how does it spread?",
                    "Are there any chemical-free alternatives I can use at home?",
                    "How do I prevent this disease next season?",
                    "Is this disease contagious to other plants nearby?",
                ],
                inputs=chat_input,
                label="💬 Quick Questions",
            )

            send_btn.click(
                fn=handle_chat,
                inputs=[chat_input, chatbot],
                outputs=[chatbot, chat_input],
            )
            chat_input.submit(
                fn=handle_chat,
                inputs=[chat_input, chatbot],
                outputs=[chatbot, chat_input],
            )

    # ── Footer ──────────────────────────────────────────────────────────────
    gr.HTML("""
        <div class="app-footer">
            🌍 <b>Plant Disease Detector</b> — Powered by AI & Sustainable Agriculture<br>
            <small>Helping farmers and gardeners protect crops with eco-friendly solutions</small>
        </div>
    """)


if __name__ == "__main__":
    app.launch()