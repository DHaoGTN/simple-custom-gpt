import os
import shutil
import urllib
import uuid
import sys

from dotenv import load_dotenv
from chat.doc_service import DocService

def chatbot():
    load_dotenv()

    HOME = os.path.dirname(os.path.abspath(__file__))

    # if os.path.exists(os.path.join(HOME, "chroma_storage")):
    #     shutil.rmtree(os.path.join(HOME, "chroma_storage"))

    os.makedirs(os.path.join(HOME, "chroma_storage"), exist_ok=True)

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

def translate():
    print("Translation...")

if len(sys.argv) < 3:
    print("Usage: python app.py -m <chat | translate>")
    sys.exit(1)

if sys.argv[1] != '-m':
    print("Usage: python app.py -m <chat | translate>")
    sys.exit(1)

mode = sys.argv[2]

if mode not in ['chat', 'translate']:
    print("Invalid mode. Please use either 'chat' or 'translate'.")
    sys.exit(1)

if mode == 'translate':
    translate()
else:
    chatbot()
