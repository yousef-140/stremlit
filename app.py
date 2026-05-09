import streamlit as st
from huggingface_hub import InferenceClient
import json
import pandas as pd

st.set_page_config(page_title="Data Extractor", page_icon="🔍", layout="centered")
st.title("🔍 Data Extractor")
st.caption("Paste any text, define fields to extract — powered by Hugging Face.")

st.sidebar.header("Settings")
hf_token = st.sidebar.text_input("HuggingFace Token", type="password", placeholder="hf_...")
st.sidebar.markdown("[Get a free token →](https://huggingface.co/settings/tokens)")

model_id = st.sidebar.selectbox(
    "Model",
    [
        "mistralai/Mistral-7B-Instruct-v0.3",
        "HuggingFaceH4/zephyr-7b-beta",
        "meta-llama/Meta-Llama-3-8B-Instruct",
    ]
)

text_input = st.text_area(
    "Paste your text here",
    height=220,
    placeholder="e.g. John Smith, 34, lives in Austin TX. He works as a software engineer at Acme Corp..."
)

fields_input = st.text_input(
    "Fields to extract (comma-separated)",
    placeholder="e.g. name, age, city, job title, company"
)

if st.button("Extract Data", type="primary"):
    if not hf_token:
        st.error("Please enter your Hugging Face token in the sidebar.")
    elif not text_input.strip():
        st.warning("Please paste some text first.")
    elif not fields_input.strip():
        st.warning("Please specify at least one field to extract.")
    else:
        fields = [f.strip() for f in fields_input.split(",") if f.strip()]
        fields_list = ", ".join(f'"{f}"' for f in fields)

        prompt = f"""<s>[INST] You are a data extraction assistant. Extract the following fields from the text.
Return ONLY a valid JSON array of objects. Each object must have these exact keys: {fields_list}.
If a field is not found, use null. If multiple records exist, return one object per record.
Do not add any explanation, only return the JSON array.

Text:
{text_input}
[/INST]"""

        with st.spinner(f"Extracting with {model_id.split('/')[1]}..."):
            try:
                client = InferenceClient(model=model_id, token=hf_token)

                response = client.text_generation(
                    prompt,
                    max_new_tokens=1024,
                    temperature=0.1,
                    return_full_text=False,
                )

                raw = response.strip()

                # Strip markdown fences if present
                if "```" in raw:
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                raw = raw.strip()

                # Find JSON array in response
                start = raw.find("[")
                end = raw.rfind("]") + 1
                if start != -1 and end > start:
                    raw = raw[start:end]

                data = json.loads(raw)

                if isinstance(data, dict):
                    data = [data]

                df = pd.DataFrame(data)

                st.success(f" Extracted {len(df)} record(s) across {len(df.columns)} field(s).")
                st.dataframe(df, use_container_width=True)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇ Download CSV",
                    data=csv,
                    file_name="extracted_data.csv",
                    mime="text/csv"
                )

                with st.expander("Raw JSON"):
                    st.json(data)

            except json.JSONDecodeError:
                st.error("Model returned unexpected format. Try a different model or simplify your fields.")
                st.code(raw)
            except Exception as e:
                st.error(f"Error: {e}")
