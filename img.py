import streamlit as st
from huggingface_hub import InferenceClient
import io

st.set_page_config(page_title="Image Generator", page_icon="🎨")
st.title("🎨 Image Generator")

client = InferenceClient(token=st.secrets["HF_TOKEN"])

prompt = st.text_input("Describe your image", placeholder="a cat sitting on the moon, photorealistic")

if st.button("Generate", type="primary"):
    if not prompt.strip():
        st.warning("Please enter a description.")
    else:
        with st.spinner("Generating..."):
            try:
                image = client.text_to_image(
                    prompt,
                    model="black-forest-labs/FLUX.1-schnell"
                )
                st.image(image, use_container_width=True)

                # Download button
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                st.download_button("⬇️ Download", buf.getvalue(), "image.png", "image/png")

            except Exception as e:
                st.error(f"Error: {e}")
