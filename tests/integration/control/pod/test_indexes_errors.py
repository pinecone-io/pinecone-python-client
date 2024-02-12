import pytest
from pinecone import PineconeApiException, PineconeApiValueError


class TestCreateIndexErrorCases:
    @pytest.mark.parametrize("property_to_delete", ["name", "dimension", "spec"])
    def test_create_index_missing_required_property(self, client, create_index_params, property_to_delete):
        create_index_params.pop(property_to_delete)
        with pytest.raises(TypeError) as e:
            client.create_index(**create_index_params)
        assert property_to_delete in str(e.value)

    @pytest.mark.parametrize("property_to_delete", ["environment"])
    def test_create_index_missing_required_spec_property(self, client, create_index_params, property_to_delete):
        create_index_params["spec"]['pod'].pop(property_to_delete)
        with pytest.raises(PineconeApiException) as e:
            client.create_index(**create_index_params)
        assert property_to_delete in e.value.body



    @pytest.mark.parametrize(
        "property_name, property_value",
        [
            ("name", ""),
            ("name", "invalid-"),
            ("name", "-invalid"),
            ("name", "toolong" * 10)
        ],
    )
    def test_create_index_invalid_property_value(self, client, create_index_params, property_name, property_value):
        with pytest.raises(PineconeApiValueError) as e:
            create_index_params[property_name] = property_value
            client.create_index(**create_index_params)
        assert property_name in str(e.value)

    @pytest.mark.parametrize(
        "property_name, property_value",
        [
            ("name", "invalid_name"),
            ("name", "Invalid-Name"),
            ("name", "in.valid"),
            ("name", "in#valid"),
            ("spec", {})
        ],
    )
    def test_create_index_invalid_property_value_serverside(self, client, create_index_params, property_name, property_value):
        with pytest.raises(PineconeApiException) as e:
            create_index_params[property_name] = property_value
            client.create_index(**create_index_params)
        assert 'Bad Request' in str(e.value)
        assert e.value.status == 400

    @pytest.mark.parametrize(
        "property_name, property_value",
        [
            ("metric", ""),
            ("metric", "foo"),
            ("dimension", 0),
            ("dimension", -1),
            ("dimension", 500000),
        ],
    )
    def test_create_index_failed_client_side_validation(self, client, create_index_params, property_name, property_value):
        with pytest.raises(PineconeApiValueError) as e:
            create_index_params[property_name] = property_value
            client.create_index(**create_index_params)
        assert f"Invalid value for `{property_name}`" in str(e.value)

    @pytest.mark.parametrize(
        "property_name, property_value",
        [
            ("name", None),
            ("dimension", None),
            ("dimension", "1"),
            ("dimension", 100.12),
            ("metric", 3),
            ("metric", None)
        ],
    )
    def test_create_index_invalid_property_type(self, client, create_index_params, property_name, property_value):
        with pytest.raises(TypeError) as e:
            create_index_params[property_name] = property_value
            client.create_index(**create_index_params)
        assert 'Invalid type for variable' in str(e.value)

    @pytest.mark.parametrize(
        "property_name, property_value",
        [
            ('environment', None),
            ('environment', ''),
            ('pod_type', 'big'),
            ('metadata_config', []),
            ('metadata_config', ""),
            ('replicas', 1.25),
            # ('replicas', -1), # skip for now, bug reported
            ('replicas', '1'),
        ],
    )
    def test_create_index_invalid_spec_property(self, client, create_index_params, property_name, property_value):
        with pytest.raises(PineconeApiException) as e:
            create_index_params['spec']['pod'][property_name] = property_value
            client.create_index(**create_index_params)
        assert (property_name in str(e.value)) or (property_value in str(e.value))


    def test_create_index_with_invalid_environment(self, client, create_index_params):
        with pytest.raises(PineconeApiException) as e:
            create_index_params['spec']['pod']['environment'] = 'nonexistent'
            client.create_index(**create_index_params)
        assert e.value.status == 404
        assert 'nonexistent' in str(e.value)

    def test_create_index_with_invalid_source_collection(self, client, create_index_params):
        with pytest.raises(PineconeApiException) as e:
            create_index_params['spec']['pod']['source_collection'] = 'nonexistent'
            client.create_index(**create_index_params)
        assert e.value.status == 400
        assert 'collection' in str(e.value)
        assert 'nonexistent' in str(e.value)

class TestDescribeIndexErrorCases:
    def test_describe_nonexistent_index(self, client):
        with pytest.raises(PineconeApiException) as e:
            client.describe_index('doesnotexist')
        assert e.value.status == 404
        assert e.value.reason == 'Not Found'
        assert 'doesnotexist' in e.value.body

    def test_describe_index_no_arguments(self, client):
        with pytest.raises(TypeError) as e:
            client.describe_index()
        assert 'missing 1 required positional argument' in str(e.value)

class TestConfigureIndexErrorCases:
    def test_configure_with_no_arguments(self, client):
        with pytest.raises(TypeError) as e:
            client.configure_index()
        assert "configure_index() missing 1 required positional argument: 'name'" in str(e.value)

    def test_configure_with_nonexistent_index(self, client):
        with pytest.raises(PineconeApiException) as e:
            client.configure_index(name='nonexistent', replicas=1)
        assert e.value.status == 404
        assert e.value.reason == 'Not Found'
        assert 'nonexistent' in e.value.body

    def test_configure_with_invalid_replicas(self, client, reusabale_index):
        with pytest.raises(PineconeApiValueError) as e:
            client.configure_index(reusabale_index, replicas=-1)
        assert 'Invalid value for `replicas`' in str(e.value)

    def test_configure_with_insufficient_quota(self, client, reusabale_index):
        with pytest.raises(PineconeApiException) as e:
            client.configure_index(name=reusabale_index, replicas=200)
        assert e.value.status == 403
        assert e.value.reason == 'Forbidden'
        assert 'Increase your quota or upgrade to create more indexes.' in e.value.body

    @pytest.mark.parametrize("invalid_pod_type", ["p1.lol", "lol.x1"])
    def test_configure_with_invalid_podtype_value(self, client, reusabale_index, invalid_pod_type):
        with pytest.raises(PineconeApiException) as e:
            client.configure_index(name=reusabale_index, pod_type=invalid_pod_type)
        assert e.value.status == 400
        assert e.value.reason == 'Bad Request'
        assert 'Invalid pod type' in e.value.body
