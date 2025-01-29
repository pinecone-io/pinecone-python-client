class DeprecatedPluginError(Exception):
    def __init__(self, message):
        super().__init__(message)


def check_for_deprecated_plugins():
    try:
        from pinecone_plugins.inference import __installables__  # type: ignore

        if __installables__ is not None:
            raise DeprecatedPluginError(
                "The `pinecone-plugin-inference` package has been deprecated. The embed and rerank functionality has been incorporated into the main `pinecone` package with no need for additional plugins. Please remove the `pinecone-plugin-inference` package from your dependencies to ensure you have the most up-to-date version of these features."
            )
    except ImportError:
        # ImportError is expected if the plugin is not installed,
        # which is the good case.
        pass
