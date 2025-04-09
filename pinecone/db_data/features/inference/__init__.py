import warnings

warnings.warn(
    "The module at `pinecone.data.features.inference` has moved to `pinecone.inference`. "
    "Please update your imports from `from pinecone.data.features.inference import Inference, AsyncioInference, RerankModel, EmbedModel` "
    "to `from pinecone.inference import Inference, AsyncioInference, RerankModel, EmbedModel`. "
    "This warning will become an error in a future version of the Pinecone Python SDK.",
    DeprecationWarning,
)

from pinecone.inference import *
