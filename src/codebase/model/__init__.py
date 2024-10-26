

from .angle import AngleEmbeddingClassifier
from .amplitude import AmplitudeEmbeddingClassifier
from .reuploading import Reuploading_classifier

def get_classifier(name: str):
    match name:
        case "reuploading":
            return Reuploading_classifier
        case "angle":
            return AngleEmbeddingClassifier
        case "amplitude":
            return AmplitudeEmbeddingClassifier
        case _:
            raise ValueError(f"Unknown classifier: {name}")