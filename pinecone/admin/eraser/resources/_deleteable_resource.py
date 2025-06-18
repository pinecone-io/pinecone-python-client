import abc


class _DeleteableResource(abc.ABC):
    @abc.abstractmethod
    def name(self):
        """Get the singular form of the resource name"""
        pass

    @abc.abstractmethod
    def name_plural(self):
        """Get the plural form of the resource name"""
        pass

    @abc.abstractmethod
    def delete(self, name):
        """Delete the resource with the given name"""
        pass

    @abc.abstractmethod
    def get_state(self, name):
        """Get the state of the resource with the given name"""
        pass

    @abc.abstractmethod
    def list(self):
        """List all resources"""
        pass
