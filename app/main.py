from datetime import datetime
from pathlib import Path
from cassis import Cas
import spacy
import stanza    
import spacy_stanza
from spacy.tokens import Doc

from ariadne.classifier import Classifier
from ariadne.contrib.inception_util import create_prediction, TOKEN_TYPE
from ariadne.contrib.spacy import SpacyNerClassifier, SpacyPosClassifier
from ariadne.server import Server

stanza.download("tr")


class SpacyPosClassifier(Classifier):
    def __init__(self, model_name: str):
        super().__init__()
        self._model = spacy_stanza.load_pipeline("tr")

    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        # Extract the tokens from the CAS and create a spacy doc from it
        words = [cas.get_covered_text(cas_token) for cas_token in cas.select(TOKEN_TYPE)]

        doc = Doc(self._model.vocab, words=words)

        # Get the pos tags
        self._model.get_pipe("tok2vec")(doc)
        self._model.get_pipe("tagger")(doc)

        # For every token, extract the POS tag and create an annotation in the CAS
        for cas_token, spacy_token in zip(cas.select(TOKEN_TYPE), doc):
            prediction = create_prediction(cas, layer, feature, cas_token.begin, cas_token.end, spacy_token.tag_)
            cas.add_annotation(prediction)


server = Server()
server.add_classifier("spacy_ner", SpacyNerClassifier("en_core_web_sm"))
server.add_classifier("spacy_pos", SpacyPosClassifier("en_core_web_sm"))

server.start()