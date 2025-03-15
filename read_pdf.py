import PyPDF2

def read_pdf(file_path):
    """
    Read and extract text from a PDF file
    """
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
        return text

if __name__ == "__main__":
    pdf_path = "Section_worksheet_Week9B.pdf"
    pdf_text = read_pdf(pdf_path)
    print(pdf_text) 