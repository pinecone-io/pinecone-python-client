import os
from abc import ABC, abstractmethod
from .verify import verify_hashes

class PineconeFeature(ABC):
    @abstractmethod
    def location(self):
        """
        Should return a string representing the location of the feature.
        """
        pass

    @abstractmethod
    def target(self):
        """
        Should return a string representing the target of the feature.
        """
        pass

    @abstractmethod
    def methods(self):
        """
        Should return a dictionary of methods implemented by the feature.
        """
        pass

    def validate_generated(self):
        """
        Checks hash values to ensure generated files have not been modified.
        """
        verify_hashes(os.path.join(self.location(), 'generated', 'hashes.yaml'))
