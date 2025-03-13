from pdf2image import convert_from_path

pages = convert_from_path('beispiel1.pdf', 500)
pages += convert_from_path('beispiel2.pdf', 500)

for count, page in enumerate(pages):
    page.save(f'out{count+1}.jpg', 'JPEG')