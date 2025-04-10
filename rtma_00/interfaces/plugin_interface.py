from abc import ABC, abstractmethod

class PluginInterface(ABC):
    """
    Abstract base class for UI plugins that add new functionality.

    Methods
    -------
    register():
        Hook to register the plugin into the main application.
    """

    @abstractmethod
    def register(self, app):
        pass
