from typing import List, Tuple
from tqdm import tqdm
from multiprocessing.pool import AsyncResult
import logging
import time
import numpy as np


def fetch_async_result(
        job_list: List[Tuple[int | str, AsyncResult]],
        process_bar: tqdm | None = None,
        max_attempt: int | None = 200  # approx 100 secs
    ) -> List:
    """
    Fetch result from asynchronous job when ready.

    :param List[AsyncResult] job_list: list with asynchronous results.
    :param tqdm | None process_bar: Process bar. If none, no process bar is plotted
    :param int | None max_attempt: Maximum number of attempts to fetch result with half a second
        sleep time between each completed iteration through all remaining open jobs.
    :return List: List with results
    """
    processed_jobs = set()
    results = []
    attempts = 0
    while len(job_list) > 0:
        if attempts > max_attempt:
            logger = logging.getLogger(__name__)
            logger.warning(f"Reached limit of {max_attempt} attempts without results. Continue.")
            break
        # get first in list
        i, async_res = job_list.pop(0)
        # wait when iterated through entire job list and still unfinished jobs
        if i in processed_jobs:
            # attempt to fetch the same
            attempts += 1
            time.sleep(.5)
            processed_jobs = set()
        processed_jobs.add(i)
        # check if ready and fetch
        if async_res.ready():
            attempts = 0
            results.append((i, async_res.get()))
            if process_bar is not None:
                process_bar.update(1)
        else:
            # otherwise add to end of list.
            job_list.append((i, async_res))
    return results


def factorize(n: int) -> Tuple[int, int]:
    sq_fact = np.round(np.sqrt(n)).astype("int")
    while n % sq_fact > 0:
        sq_fact -= 1
    return sq_fact, n // sq_fact