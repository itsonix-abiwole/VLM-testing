from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

print("PyTorch Version:", torch.__version__)
print("CUDA verfügbar:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("Keine GPU erkannt.")



model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.float16, device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

messages = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text":"""Konvertiere das folgende Dokument in Markdown.
                                    Gib nur den Markdown ohne Erklärungstext zurück. Fügen Sie keine Begrenzungszeichen wie '''markdown''' oder '''html''' ein.
                                    REGELN:
                                    - Sie müssen alle Informationen auf der Seite einschließen. Schließen Sie Kopf- und Fußzeilen, Diagramme, Infografiken oder Untertexte nicht aus.
                                    - Geben Sie Tabellen in einem HTML-Format zurück.
                                    - Logos sollten in Klammern gesetzt werden. Beispiel: <logo>Coca-Cola<logo>
                                    - Wasserzeichen sollten in eckige Klammern gesetzt werden. Bsp: <watermark>OFFIZIELLE KOPIE<watermark>
                                    - Seitenzahlen sollten in Klammern eingeschlossen werden. Bsp: <page_number>14<page_number> oder <page_number>9/22<page_number>
                                    - Bevorzugen Sie die Verwendung von ☐ und ☑ für Kontrollkästchen."""}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "out1.jpg"},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to("cuda")

generated_ids = model.generate(**inputs, max_new_tokens=128)

output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print("Generierter Output:", output_text)