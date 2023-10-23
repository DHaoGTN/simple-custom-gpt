import os
import shutil
import urllib
import uuid

from dotenv import load_dotenv
from core.doc_service import DocService

load_dotenv()

HOME = os.path.dirname(os.path.abspath(__file__))

# if os.path.exists(os.path.join(HOME, "chroma_storage")):
#     shutil.rmtree(os.path.join(HOME, "chroma_storage"))

os.makedirs(os.path.join(HOME, "chroma_storage"), exist_ok=True)

# uuid_memory = str(uuid.uuid4())
collection = 'gtn-mobile'
temp_save_directory = os.path.join(HOME, 'dataset')

filename = "Mobile-rule.docx"  # add the file name here
data_path = os.path.join(temp_save_directory, filename)

cht_mdl = DocService(data_path, persist_directory=os.path.join(HOME, "chroma_storage", collection))
cht_mdl.fetch_document()
cht_mdl.create_vector_index() # -> we can skip this process from 2nd run because already exist chroma vector db with persist_directory

while True:
    query = input("input the text here: ")
    output = cht_mdl.query_document(prompt=query)
    print(output)
