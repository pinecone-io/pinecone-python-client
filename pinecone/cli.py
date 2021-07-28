#
# Copyright (c) 2020-2021 Pinecone Systems Inc. All right reserved.
#

# import fire
import configparser
import pathlib

__all__ = ["CLI"]


class CLI:
    def init(self, api_key: str = None):
        """Configures the Pinecone client.

        Usage:

        .. code-block:: bash

            pinecone init --api_key=YOUR_API_KEY

        :param api_key: your Pinecone API key
        """
        # Construct config
        config = configparser.ConfigParser()
        config["default"] = {
            "api_key": api_key,
        }
        config["default"] = {key: val for key, val in config["default"].items() if val}
        # Write config file
        with pathlib.Path.home().joinpath(".pinecone").open("w") as configfile:
            config.write(configfile)


def main():
    raise NotImplementedError("this method has been removed")
    # try:
    #     fire.Fire(CLI)
    # except KeyboardInterrupt:
    #     pass


if __name__ == "__main__":
    main()
