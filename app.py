import streamlit as st
from huggingface_hub import InferenceClient
import json
import pandas as pd

st.set_page_config(page_title="Data Extractor", page_icon="🔍")
st.title("🔍 Data Extractor")

client = InferenceClient(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    token=st.secrets["HF_TOKEN"]
)

text_input = st.text_area("Paste your text here", height=200)
fields_input = st.text_input("Fields to extract (comma-separated)", placeholder="e.g. name, age, city, company")

if st.button("Extract", type="primary"):
    if not text_input.strip() or not fields_input.strip():
        st.warning("Please fill in both fields.")
    else:
        fields = [f.strip() for f in fields_input.split(",") if f.strip()]
        fields_list = ", ".join(f'"{f}"' for f in fields)

        with st.spinner("Extracting..."):
            try:
                response = client.chat_completion(
                    messages=[
                        {
                            "role": "system",
                            "content": f"You are a data extraction assistant. Extract fields from text and return ONLY a valid JSON array of objects with keys: {fields_list}. Use null for missing fields. No explanation, just JSON."
                        },
                        {
                            "role": "user",
                            "content": text_input
                        }
                    ],
                    max_tokens=1024,
                    temperature=0.1,
                )

                raw = response.choices[0].message.content.strip()

                # Clean markdown fences
                if "```" in raw:
                    raw = raw.split("```")[1].lstrip("json").strip()

                # Find JSON array
                start, end = raw.find("["), raw.rfind("]") + 1
                if start != -1 and end > start:
                    raw = raw[start:end]

                data = json.loads(raw)
                if isinstance(data, dict):
                    data = [data]

                df = pd.DataFrame(data)
                st.success(f"Extracted {len(df)} record(s)")
                st.dataframe(df, use_container_width=True)

                st.download_button(
                    "⬇️ Download CSV",
                    df.to_csv(index=False).encode("utf-8"),
                    "extracted_data.csv",
                    "text/csv"
                )

            except json.JSONDecodeError:
                st.error("Could not parse model output as JSON.")
                st.code(raw)
            except Exception as e:
                st.error(f"Error: {e}")
