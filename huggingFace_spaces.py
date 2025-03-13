from gradio_client import Client, handle_file
from dotenv import load_dotenv
import os

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

prompt = """Konvertiere das folgende Dokument in Markdown.
            Gib nur den Markdown ohne Erklärungstext zurück. 
            Füge keine Begrenzungszeichen wie '''markdown''' oder '''html''' ein.
            Alle Informationen, wie Kopf- und Fußzeilen, Diagramme, Infografiken oder Untertexte sollen konvertiert werden.
            Gib Tabellen in einem HTML-Format zurück."""

image_url = os.getenv("IMAGE_URL1")
image_path = 'resources/out1.jpg'


# InternVL2_5-8B
client = Client("developer0hye/InternVL2_5-8B", hf_token=hf_token)
result = client.predict(
		media_input=handle_file(image_path),
		text_input=prompt,
		api_name="/internvl_inference"
)


# Qwen2.5-VL-7B-Instruct (GPU thorwos an error)
# client = Client("prithivMLmods/Qwen2.5-VL-7B-Instruct", hf_token=hf_token)
# result = client.predict(
# 		message={"text":prompt,"files":[handle_file(image_path)]},
# 		api_name="/chat"
# )


print(result)