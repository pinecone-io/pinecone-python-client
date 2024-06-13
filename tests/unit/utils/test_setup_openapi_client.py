import re

import pytest

from pinecone.config import ConfigBuilder
from pinecone.core.client.api.manage_indexes_api import ManageIndexesApi
from pinecone.core.client.api_client import ApiClient
from pinecone.utils.setup_openapi_client import setup_openapi_client, build_plugin_setup_client


class TestSetupOpenAPIClient():
    def test_setup_openapi_client(self):
        config = ConfigBuilder.build(
            api_key="my-api-key",
            host="https://my-controller-host"
        )
        openapi_config = ConfigBuilder.build_openapi_config(config)
        assert openapi_config.host == "https://my-controller-host"

        control_plane_client = setup_openapi_client(ApiClient, ManageIndexesApi, config=config, openapi_config=openapi_config, pool_threads=2)
        user_agent_regex = re.compile(r"python-client-\\d+\\.\\d+\\.\\d+([a-z]+\\d+)? \\(urllib3\\:\\d+\\.\\d+\\.\\d+\\)")
        assert re.match(user_agent_regex, control_plane_client.api_client.user_agent)
        assert re.match(user_agent_regex, control_plane_client.api_client.default_headers['User-Agent'])

    def test_setup_openapi_client_with_api_version(self):
        config = ConfigBuilder.build(
            api_key="my-api-key",
            host="https://my-controller-host",
        )
        openapi_config = ConfigBuilder.build_openapi_config(config)
        assert openapi_config.host == "https://my-controller-host"

        control_plane_client = setup_openapi_client(ApiClient, ManageIndexesApi, config=config, openapi_config=openapi_config, pool_threads=2, api_version="2024-07")
        user_agent_regex = re.compile(r"python-client-\\d+\\.\\d+\\.\\d+([a-z]+\\d+)? \\(urllib3\\:\\d+\\.\\d+\\.\\d+\\)")
        assert re.match(user_agent_regex, control_plane_client.api_client.user_agent)
        assert re.match(user_agent_regex, control_plane_client.api_client.default_headers['User-Agent'])
        assert control_plane_client.api_client.default_headers['X-Pinecone-API-Version'] == "2024-07"


class TestBuildPluginSetupClient():
    @pytest.mark.parametrize("plugin_api_version,plugin_host", [
        (None, None),
        ("2024-07", "https://my-plugin-host")
    ])
    def test_setup_openapi_client_with_host_override(self, plugin_api_version, plugin_host):
        # These configurations represent the configurations that the core sdk
        # (e.g. Pinecone class) will have built prior to invoking the plugin setup.
        # In real usage, this takes place during the Pinecone class initialization 
        # and pulls together configuration from all sources (kwargs and env vars).
        # It reflects a merging of the user's configuration and the defaults set 
        # by the sdk.
        config = ConfigBuilder.build(
            api_key="my-api-key",
            host="https://api.pinecone.io",
            source_tag="my_source_tag",
            proxy_url="http://my-proxy.com",
            ssl_ca_certs="path/to/bundle.pem"
        )
        openapi_config = ConfigBuilder.build_openapi_config(config)

        # The core sdk (e.g. Pinecone class) will be responsible for invoking the
        # build_plugin_setup_client method before passing the result to the plugin
        # install method. This is 
        # somewhat like currying the openapi setup function, because we want some
        # information to be controled by the core sdk (e.g. the user-agent string, 
        # proxy settings, etc) while allowing the plugin to pass the parts of the 
        # configuration that are relevant to it such as api version, base url if 
        # served from somewhere besides api.pinecone.io, etc.
        client_builder = build_plugin_setup_client(config=config, openapi_config=openapi_config, pool_threads=2)

        # The plugin machinery in pinecone_plugin_interface will be the one to call
        # this client_builder function using classes and other config it discovers inside the 
        # pinecone_plugin namespace package. Putting plugin configuration and references
        # to the implementation classes into a spot where the pinecone_plugin_interface
        # can find them is the responsibility of the plugin developer.
        #
        # Passing ManagedIndexesApi and ApiClient here are just a standin for testing 
        # purposes; in a real plugin, the class would be something else related 
        # to a new feature, but to test that this setup works I just need a FooApi 
        # class generated off the openapi spec.
        plugin_api = ManageIndexesApi
        plugin_client = client_builder(
            api_client_klass=ApiClient,
            api_klass=plugin_api,
            api_version=plugin_api_version,
            host=plugin_host
        )

        # Returned client is an instance of the input class
        assert isinstance(plugin_client, plugin_api)

        # We want requests from plugins to have a user-agent matching the host SDK.
        user_agent_regex = re.compile(r"python-client-\d+\.\d+\.\d+ \(urllib3\:\d+\.\d+\.\d+\)")
        assert re.match(user_agent_regex, plugin_client.api_client.user_agent)
        assert re.match(user_agent_regex, plugin_client.api_client.default_headers['User-Agent'])

        # User agent still contains the source tag that was set in the sdk config
        assert 'my_source_tag' in plugin_client.api_client.default_headers['User-Agent']

        # Proxy settings should be passed from the core sdk to the plugin client
        assert plugin_client.api_client.configuration.proxy == "http://my-proxy.com"
        assert plugin_client.api_client.configuration.ssl_ca_cert == "path/to/bundle.pem"

        # Plugins need to be able to pass their own API version (optionally)
        assert plugin_client.api_client.default_headers.get('X-Pinecone-API-Version') == plugin_api_version

        # Plugins need to be able to override the host (optionally)
        if plugin_host:
            assert plugin_client.api_client.configuration._base_path == plugin_host
        else:
            # When plugin does not set a host, it should default to the host set in the core sdk
            assert plugin_client.api_client.configuration._base_path == "https://api.pinecone.io"
