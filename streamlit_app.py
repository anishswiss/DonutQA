import base64
import io
import os

import requests
import streamlit as st
from PIL import Image


st.set_page_config(page_title="Document QA", page_icon="📄")
st.title("Document Question Answering")
st.write(
    "Upload a document image, ask questions, and get answers using AI-powered "
    "document understanding."
)


def encode_image(file_bytes: bytes) -> str:
    """Normalize to JPEG (RGB format) and return base64 string."""
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# Document type configuration
# Endpoints are stored as environment variables: DOCUMENT_TYPE_ENDPOINT_URI and DOCUMENT_TYPE_ENDPOINT_KEY
# Example: PAYSTUB_ENDPOINT_URI, PAYSTUB_ENDPOINT_KEY, W2_ENDPOINT_URI, W2_ENDPOINT_KEY
DOCUMENT_TYPES = {
    "Paystub": {
        "uri_env": "PAYSTUB_ENDPOINT_URI",
        "key_env": "PAYSTUB_ENDPOINT_KEY",
    },
    "W2": {
        "uri_env": "W2_ENDPOINT_URI",
        "key_env": "W2_ENDPOINT_KEY",
    },
}

# Add document type selector
document_type = st.selectbox(
    "Select document type",
    options=list(DOCUMENT_TYPES.keys()),
    help="Choose the type of document you want to analyze."
)

# Get endpoint configuration for selected document type
doc_config = DOCUMENT_TYPES[document_type]
scoring_uri = os.getenv(doc_config["uri_env"], "")
endpoint_key = os.getenv(doc_config["key_env"], "")

question_mode = st.radio(
    "Question mode",
    ["Single question", "Multiple questions"],
    horizontal=True
)

if question_mode == "Single question":
    question = st.text_input("Question", value="What is the net pay?")
    questions = [question] if question else []
else:
    questions_text = st.text_area(
        "Questions (one per line)",
        value="What is the net pay?\nWhat is the gross pay?\nWhat is the employee name?",
        height=150,
        help="Enter one question per line. All questions will be processed together."
    )
    questions = [q.strip() for q in questions_text.split("\n") if q.strip()]

uploaded_file = st.file_uploader("Upload document image", type=["png", "jpg", "jpeg"])

col1, col2 = st.columns(2)
with col1:
    run_button = st.button("Get answer", type="primary")
with col2:
    use_sample = st.button(f"Use sample {document_type.lower()}")

if use_sample:
    sample_path = os.path.join(f"test_{document_type.lower()}.jpg")
    try:
        with open(sample_path, "rb") as sample_file:
            uploaded_file = io.BytesIO(sample_file.read())
            uploaded_file.name = os.path.basename(sample_path)
        st.success(f"Loaded bundled sample {document_type.lower()}.")
    except FileNotFoundError:
        st.error(f"Sample {document_type.lower()} not found in project root.")

if run_button:
    if not scoring_uri:
        st.error(f"Endpoint URI not configured for {document_type}. Please set {doc_config['uri_env']} environment variable.")
    elif not endpoint_key:
        st.error(f"Endpoint key not configured for {document_type}. Please set {doc_config['key_env']} environment variable.")
    elif not uploaded_file:
        st.error("Please upload an image.")
    elif not questions:
        st.error("Please provide at least one question.")
    else:
        if hasattr(uploaded_file, "getvalue"):
            image_bytes = uploaded_file.getvalue()
        else:
            image_bytes = uploaded_file.read()

        try:
            encoded_image = encode_image(image_bytes)
        except Exception as exc:
            st.error(f"Unable to read image: {exc}")
            st.stop()

        payload = {"image": encoded_image, "questions": questions}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {endpoint_key}",
        }

        import time
        start_time = time.time()
        num_questions = len(questions)
        spinner_text = f"Querying Azure ML endpoint with {num_questions} question{'s' if num_questions > 1 else ''} (this may take 20-30 seconds per question)..."
        with st.spinner(spinner_text):
            try:
                response = requests.post(
                    scoring_uri, json=payload, headers=headers, timeout=120 * num_questions
                )
                elapsed = time.time() - start_time
                st.info(f"Request completed in {elapsed:.1f} seconds")
                
                response.raise_for_status()
                result = response.json()
                answers = result.get("answers", [])
                
                if answers:
                    if len(answers) == 1:
                        st.success(f"**Answer:** {answers[0]}")
                    else:
                        st.success("**Answers:**")
                        for i, (q, a) in enumerate(zip(questions, answers), 1):
                            st.write(f"**Q{i}:** {q}")
                            st.write(f"**A{i}:** {a}")
                            if i < len(answers):
                                st.divider()
                else:
                    # Fallback for single answer format (backward compatibility)
                    answer = result.get("answer")
                    if answer:
                        st.success(f"**Answer:** {answer}")
                    else:
                        st.warning(f"Endpoint response: {result}")
            except requests.Timeout:
                elapsed = time.time() - start_time
                st.error(
                    f"Request timed out after {elapsed:.1f} seconds. "
                    "The backend may still be processing. "
                    "Check Azure ML endpoint logs to see if the request completed."
                )
            except requests.HTTPError as http_err:
                elapsed = time.time() - start_time
                error_text = response.text if hasattr(response, 'text') else str(http_err)
                status_code = response.status_code if hasattr(response, 'status_code') else 'Unknown'
                
                if status_code == 408:
                    st.warning(
                        f"⚠️ Gateway timeout (408) after {elapsed:.1f} seconds, but backend may have completed successfully.\n\n"
                        f"**Check Azure ML logs** - if you see 'run(): Returning result' in the logs, "
                        f"the model processed successfully but the response didn't reach Streamlit.\n\n"
                        f"Error details: {error_text}"
                    )
                else:
                    st.error(
                        f"Endpoint returned {status_code} after {elapsed:.1f} seconds: {error_text}\n\n"
                        f"Check Azure ML endpoint logs to see if the request reached the backend."
                    )
            except requests.RequestException as req_err:
                elapsed = time.time() - start_time
                st.error(f"Request failed after {elapsed:.1f} seconds: {req_err}")

        with st.expander("Request payload (debug)"):
            st.json({"questions": questions, "num_questions": len(questions), "image_bytes": len(encoded_image)})

        if uploaded_file:
            st.image(image_bytes, caption=uploaded_file.name, use_column_width=True)

