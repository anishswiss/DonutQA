import base64
import io
import os

import requests
import streamlit as st
from PIL import Image


st.set_page_config(page_title="Donut QA", page_icon="📄")
st.title("Donut QA – Document Question Answering")
st.write(
    "Upload a pay stub image, ask a question, and the app will call the Azure ML "
    "endpoint that hosts your Donut model."
)


def encode_image(file_bytes: bytes) -> str:
    """Normalize to JPEG (Donut expects RGB) and return base64 string."""
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


scoring_uri = st.text_input(
    "Endpoint scoring URI",
    value=os.getenv("DONUT_ENDPOINT_URI", ""),
    placeholder="https://<endpoint>.<region>.inference.ml.azure.com/score",
)

endpoint_key = st.text_input(
    "Endpoint key",
    value=os.getenv("DONUT_ENDPOINT_KEY", ""),
    type="password",
    help="Use the primary or secondary key from the Azure ML endpoint.",
)

question = st.text_input("Question", value="What is the net pay?")
uploaded_file = st.file_uploader("Document image", type=["png", "jpg", "jpeg"])

col1, col2 = st.columns(2)
with col1:
    run_button = st.button("Get answer", type="primary")
with col2:
    use_sample = st.button("Use sample pay stub")

if use_sample:
    sample_path = os.path.join("test_pay_stub.jpg")
    try:
        with open(sample_path, "rb") as sample_file:
            uploaded_file = io.BytesIO(sample_file.read())
            uploaded_file.name = os.path.basename(sample_path)
        st.success("Loaded bundled sample pay stub.")
    except FileNotFoundError:
        st.error("Sample pay stub not found in project root.")

if run_button:
    if not scoring_uri:
        st.error("Please provide the endpoint scoring URI.")
    elif not endpoint_key:
        st.error("Please provide the endpoint key.")
    elif not uploaded_file:
        st.error("Please upload an image.")
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

        payload = {"image": encoded_image, "question": question}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {endpoint_key}",
        }

        import time
        start_time = time.time()
        with st.spinner("Querying Azure ML endpoint (this may take 20-30 seconds)..."):
            try:
                response = requests.post(
                    scoring_uri, json=payload, headers=headers, timeout=120
                )
                elapsed = time.time() - start_time
                st.info(f"Request completed in {elapsed:.1f} seconds")
                
                response.raise_for_status()
                result = response.json()
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
            st.json({"question": question, "image_bytes": len(encoded_image)})

        if uploaded_file:
            st.image(image_bytes, caption=uploaded_file.name, use_column_width=True)

