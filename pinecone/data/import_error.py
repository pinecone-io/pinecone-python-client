class IndexClientInstantiationError(Exception):
    def __init__(self, index_args, index_kwargs):
        formatted_args = ", ".join(map(repr, index_args))
        formatted_kwargs = ", ".join(f"{key}={repr(value)}" for key, value in index_kwargs.items())
        combined_args = ", ".join([a for a in [formatted_args, formatted_kwargs] if a.strip()])

        self.message = f"""You are attempting to access the Index client directly from the pinecone module. The Index client must be instantiated through the parent Pinecone client instance so that it can inherit shared configurations such as API key.

    INCORRECT USAGE:
        ```
        import pinecone

        pc = pinecone.Pinecone(api_key='your-api-key')
        index = pinecone.Index({combined_args})
        ```

    CORRECT USAGE:
        ```
        from pinecone import Pinecone

        pc = Pinecone(api_key='your-api-key')
        index = pc.Index({combined_args})
        ```
        """
        super().__init__(self.message)


class Index:
    def __init__(self, *args, **kwargs):
        raise IndexClientInstantiationError(args, kwargs)
