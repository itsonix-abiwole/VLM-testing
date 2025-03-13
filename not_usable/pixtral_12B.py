# download the model 
from huggingface_hub import snapshot_download
from pathlib import Path

from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage, TextChunk, ImageURLChunk
from mistral_common.protocol.instruct.request import ChatCompletionRequest

mistral_models_path = Path.home().joinpath('mistral_models', 'Pixtral')
mistral_models_path.mkdir(parents=True, exist_ok=True)

snapshot_download(repo_id="mistralai/Pixtral-12B-2409", allow_patterns=["params.json", "consolidated.safetensors", "tekken.json"], local_dir=mistral_models_path)

tokenizer = MistralTokenizer.from_file(f"{mistral_models_path}/tekken.json")
model = Transformer.from_folder(mistral_models_path)

url = "https://drive.google.com/drive/folders/19LT_8PyCDq00NeHJGrDPHY9UUxA2D89H"
prompt = """Konvertiere das folgende Dokument in Markdown.
            Gib nur den Markdown ohne Erklärungstext zurück. Fügen Sie keine Begrenzungszeichen wie '''markdown''' oder '''html''' ein.
            REGELN:
            - Sie müssen alle Informationen auf der Seite einschließen. Schließen Sie Kopf- und Fußzeilen, Diagramme, Infografiken oder Untertexte nicht aus.
            - Geben Sie Tabellen in einem HTML-Format zurück.
            - Logos sollten in Klammern gesetzt werden. Beispiel: <logo>Coca-Cola<logo>
            - Wasserzeichen sollten in eckige Klammern gesetzt werden. Bsp: <watermark>OFFIZIELLE KOPIE<watermark>
            - Seitenzahlen sollten in Klammern eingeschlossen werden. Bsp: <page_number>14<page_number> oder <page_number>9/22<page_number>
            - Bevorzugen Sie die Verwendung von ☐ und ☑ für Kontrollkästchen."""

completion_request = ChatCompletionRequest(messages=[UserMessage(content=[ImageURLChunk(image_url=url), TextChunk(text=prompt)])])

encoded = tokenizer.encode_chat_completion(completion_request)

images = encoded.images
tokens = encoded.tokens

out_tokens, _ = generate([tokens], model, images=[images], max_tokens=256, temperature=0.35, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
result = tokenizer.decode(out_tokens[0])

print(result)