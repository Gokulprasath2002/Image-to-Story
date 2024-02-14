import re
from dotenv import find_dotenv, load_dotenv
import streamlit as st
from transformers import pipeline
from langchain_community.llms import HuggingFaceHub

class ImageToStoryApp:
    def __init__(self):
        load_dotenv(find_dotenv())
        self.image_to_text_pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
        self.llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature": 0.6})

    def image_to_text(self, image_path):
        try:
            text = self.image_to_text_pipeline(image_path)
            return text
        except Exception as e:
            st.error(f"Error occurred while processing image: {e}")
            return None

    def generate_story(self, text):
        try:
            prompt = f"{text[0].get('generated_text')}, Create a heartwarming story about the image."
            response = self.llm(prompt, max_length=100, num_return_sequences=1, do_sample=True, top_k=50, top_p=0.95, temperature=0.6, return_full_text=True)
            return response
        except Exception as e:
            st.error(f"Error occurred while generating story: {e}")
            return None

    def run(self):
        st.set_page_config(page_title="Image to Story", page_icon="📚", layout="wide")
        st.header("Image to Story")
        
        uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            bytes_data = uploaded_file.getvalue()
            with open(uploaded_file.name, "wb") as f:
                f.write(bytes_data)

            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            scenario = self.image_to_text(uploaded_file.name)

            if scenario:
                story = self.generate_story(scenario)
                if story:
                    pattern = r"^.*?Create a heartwarming story about the image\.\s+"
                    result = re.sub(pattern, "", story, flags=re.S)
                    story_text = result

                    with st.expander("Scenario"):
                        st.write(scenario[0].get('generated_text'))
                    with st.expander("Story"):
                        st.write(story_text)

if __name__ == "__main__":
    app = ImageToStoryApp()
    app.run()