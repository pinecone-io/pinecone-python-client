import warnings

# Display a warning for old imports
warnings.warn(
    "The module at `pinecone.data.features.inference` has moved to `pinecone.inference`. "
    "Please update your imports from `from pinecone.data.features.inference import Inference, AsyncioInference, RerankModel, EmbedModel` "
    "to `from pinecone.inference import Inference, AsyncioInference, RerankModel, EmbedModel`. "
    "This warning will become an error in a future version of the Pinecone Python SDK.",
    DeprecationWarning,
)

# Import from the new location to maintain backward compatibility
from pinecone.inference import Inference, AsyncioInference, RerankModel, EmbedModel
