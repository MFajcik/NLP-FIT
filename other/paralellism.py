import os
from concurrent.futures import ProcessPoolExecutor

from preprocessing.nlp_io import read_word_chunks
from preprocessing.tools import find_textfile_split_points

DEFAULT_CHUNK_SIZE = 5242880  # 5MB


def _worker(callback, file, idx, start_offset, end_offset, opts, logger, chunk_size=int(DEFAULT_CHUNK_SIZE)):
    result = None
    for textchunk in read_word_chunks(file, chunk_size, start_offset, end_offset):
        result = callback(textchunk, result, idx, opts, logger)
    return result


def parallel_worker(callback, file, opts, logger, num_of_processes=8, worker=_worker, chunk_size=DEFAULT_CHUNK_SIZE,
                    pool_executor=ProcessPoolExecutor):
    filesize = os.path.getsize(file)
    logger.info("File size %d" % filesize)
    offsets = find_textfile_split_points(file, num_of_processes)
    # append end of file
    offsets.append(filesize)
    futures = [None] * num_of_processes
    with pool_executor(max_workers=num_of_processes) as p:
        for idx in list(range(num_of_processes)):
            start_offset = offsets[idx]
            end_offset = offsets[idx + 1]
            if num_of_processes == 1:
                return worker(callback, file, idx, start_offset, end_offset, opts, logger, chunk_size=chunk_size)
            futures[idx] = p.submit(worker, callback, file, idx, start_offset, end_offset, opts, logger,
                                    chunk_size=chunk_size)
    return futures
