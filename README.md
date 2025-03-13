# VLM-testing
Testing VLMs via [https://huggingface.co/spaces](HuggingFace Spaces).


##### For usage create an `.env` file and add:
- your [https://huggingface.co/docs/hub/security-tokens](HuffgingFace access token) as `HF_TOKEN`


##### Providing a PDF/image
1. If you have a PDF, convert it to an image first with the `pdf_to_image.py`script
2. Provide the image
    - If you are using an **online hosted image**, store the url of the image in the `.env` file as `IMAGE_URL`
    - If you are using a **local image**, store it in resources and provide the path to it in the `huggingFace_spaces.py`script
