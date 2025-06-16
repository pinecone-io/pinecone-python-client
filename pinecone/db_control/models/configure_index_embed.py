from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ConfigureIndexEmbed:
    """Configuration for embedding settings when configuring an index.

    Attributes:
        model (str): The name of the embedding model to use with the index.
        field_map (Dict[str, str]): Identifies the name of the text field from your document model that will be embedded.
        read_parameters (Optional[Dict[str, Any]]): The read parameters for the embedding model.
        write_parameters (Optional[Dict[str, Any]]): The write parameters for the embedding model.
    """

    model: str
    field_map: Dict[str, str]
    read_parameters: Optional[Dict[str, Any]] = None
    write_parameters: Optional[Dict[str, Any]] = None

    def as_dict(self) -> Dict[str, Any]:
        """Convert the ConfigureIndexEmbed instance to a dictionary.

        Returns:
            Dict[str, Any]: A dictionary representation of the ConfigureIndexEmbed instance.
        """
        return {
            "model": self.model,
            "field_map": self.field_map,
            "read_parameters": self.read_parameters,
            "write_parameters": self.write_parameters,
        }
