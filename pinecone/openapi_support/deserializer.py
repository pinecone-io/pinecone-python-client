import json


from .model_utils import deserialize_file, file_type, validate_and_convert_types


class Deserializer:
    @staticmethod
    def deserialize(response, response_type, config, _check_type):
        """Deserializes response into an object.

        :param response: RESTResponse object to be deserialized.
        :param response_type: For the response, a tuple containing:
            valid classes
            a list containing valid classes (for list schemas)
            a dict containing a tuple of valid classes as the value
            Example values:
            (str,)
            (Pet,)
            (float, none_type)
            ([int, none_type],)
            ({str: (bool, str, int, float, date, datetime, str, none_type)},)
        :param _check_type: boolean, whether to check the types of the data
            received from the server
        :type _check_type: bool

        :return: deserialized object.
        """
        # handle file downloading
        # save response body into a tmp file and return the instance
        if response_type == (file_type,):
            content_disposition = response.getheader("Content-Disposition")
            return deserialize_file(response.data, config, content_disposition=content_disposition)

        # fetch data from response object
        try:
            received_data = json.loads(response.data)
        except ValueError:
            received_data = response.data

        # store our data under the key of 'received_data' so users have some
        # context if they are deserializing a string and the data type is wrong

        deserialized_data = validate_and_convert_types(
            received_data, response_type, ["received_data"], True, _check_type, configuration=config
        )
        return deserialized_data
