from langchain.document_loaders import PyPDFLoader
from chat.base_service import ChatService


class PDFService(ChatService):

    def fetch_document(self):
        self.saved_file_path.endswith(["doc", "docx"])
        self.loaders = PyPDFLoader(self.saved_file_path)
