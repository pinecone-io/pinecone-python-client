import asyncio
from typing import Dict, Any
from loguru import logger
import traceback


def asyncio_exception_handler(loop: asyncio.AbstractEventLoop, context: Dict[str, Any]):
    # context["message"] will always be there; but context["exception"] may not
    logger.error("Unhandled exception caught by event loop exception handler: {}", context['message'])

    if "exception" in context:
        _e: Exception = context.get('exception')
        if _e and _e.__traceback__:
            traceback.print_tb(_e.__traceback__)
            logger.error(f"Caught exception: {_e}")
        else:
            logger.error("No traceback information available.")
    else:
        logger.error("No exception available in context.")

    logger.info("Restarting process.")
    loop.stop()


def safe_set_result(result_future: asyncio.Future, result):
    if not result_future.cancelled():
        result_future.set_result(result)
