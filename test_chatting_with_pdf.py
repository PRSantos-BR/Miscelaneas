from typing import Any, Dict, List, Optional

from numpy import vectorize
from qdrant_client import QdrantClient, models  # Import Cliente

from langchain_community.vectorstores.qdrant import Qdrant
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms.ollama import Ollama

from speech_recognition import Microphone, AudioData, Recognizer
from speech_recognition import exceptions as sr_exceptions

import pyttsx3

from deep_translator import GoogleTranslator


def speech_recognition() -> str:
    microfone = Recognizer()
    with Microphone() as source:
        microfone.adjust_for_ambient_noise(source)

        print("Diga alguma coisa: ")

        audio: AudioData = microfone.listen(source)

        try:
            frase = microfone.recognize_google(audio_data=audio,
                                               language='pt-BR')
            print('Você disse: ' + frase)
        except sr_exceptions:
            print('Não entendi o que você disse!')

        return frase


def talking(texto: str) -> None:
    engine = pyttsx3.init()

    voices = engine.getProperty('voices')
    engine.setProperty(name='rate', value=150)
    engine.setProperty(name='volume', value=10)
    engine.setProperty(name='voice', value=voices[0].id)
    engine.say(texto)
    engine.runAndWait()


colletion_name = 'documentos_claro_geodata_llama2_7b'
model_name = 'llama2:7b'

# Configura o cliente
client_qdrant: QdrantClient = QdrantClient(location='localhost',
                                           port=6333)
#
# Create a colletion in vector store (QDrant)
vector_store_qdrant: Qdrant = Qdrant(client=client_qdrant,
                                     collection_name=colletion_name,
                                     embeddings=OllamaEmbeddings(model=model_name))

#
qa: Any = RetrievalQA.from_chain_type(llm=Ollama(base_url='http://localhost:11434',
                                                 model=model_name),
                                      chain_type='stuff',
                                      retriever=vector_store_qdrant.as_retriever())

while True:
    #  query: str = speech_recognition()
    query: str = 'Interação ...'
    print('\n\n')
    response = qa.invoke(query)

    tradutor = GoogleTranslator(source='en',
                                target='pt')

    print(tradutor.translate(response['result']))
    talking(tradutor.translate(response['result']))
