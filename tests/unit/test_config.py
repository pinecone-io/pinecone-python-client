import pinecone
from pinecone.config import logger, Config


def test_update_log_level():
    default_level = 'ERROR'
    assert list(logger._core.handlers.values())[0].levelno == logger.level(default_level).no
    new_level = 'INFO'
    pinecone.init(log_level=new_level)
    assert list(logger._core.handlers.values())[0].levelno == logger.level(new_level).no


def test_multi_init():
    env = 'test-env'
    level = 'INFO'
    # first init() sets log level
    pinecone.init(log_level=level)
    assert Config.ENVIRONMENT == 'us-west1-gcp'
    assert list(logger._core.handlers.values())[0].levelno == logger.level(level).no
    # next init() shouldn't clobber log level
    pinecone.init(environment=env)
    assert Config.ENVIRONMENT == env
    assert list(logger._core.handlers.values())[0].levelno == logger.level(level).no
