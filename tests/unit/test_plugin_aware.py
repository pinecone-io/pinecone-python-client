import pytest
from pinecone.utils.plugin_aware import PluginAware
from pinecone.config import Config
from pinecone.openapi_support.configuration import Configuration as OpenApiConfig


class TestPluginAware:
    def test_errors_when_required_attributes_are_missing(self):
        class Foo(PluginAware):
            def __init__(self):
                # does not set config, openapi_config, or pool_threads
                super().__init__()

        with pytest.raises(AttributeError) as e:
            Foo()

        assert "config" in str(e.value)
        assert "openapi_config" in str(e.value)
        assert "pool_threads" in str(e.value)

    def test_correctly_raise_attribute_errors(self):
        class Foo(PluginAware):
            def __init__(self):
                self.config = Config()
                self.openapi_config = OpenApiConfig()
                self.pool_threads = 1

                super().__init__()

        foo = Foo()

        with pytest.raises(AttributeError) as e:
            foo.bar()

        assert "bar" in str(e.value)

    def test_plugins_are_lazily_loaded(self):
        class Pinecone(PluginAware):
            def __init__(self):
                self.config = Config()
                self.openapi_config = OpenApiConfig()
                self.pool_threads = 10

                super().__init__()

        pc = Pinecone()
        assert "assistant" not in dir(pc)

        assert pc.assistant is not None
