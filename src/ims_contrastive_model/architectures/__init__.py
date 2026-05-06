from .base import IMSBaseAutoencoderArchitecture
from .ContrastiveAutoencoderSkrajny import ContrastiveAutoencoderSkrajny

# TODO list of import
ARCHITECTURES_REGISTRY = {
    "ContrastiveAutoencoderSkrajny": ContrastiveAutoencoderSkrajny
    }