import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from preprocessing.io import read_word_chunks_with_offset
from preprocessing.tools import find_textfile_split_points


def _worker(callback, file, idx, start_offset, end_offset, opts, logger, chunk_size=int(10485760/2)):
    result = None
    for textchunk in read_word_chunks_with_offset(file, chunk_size, start_offset, end_offset):
        result = callback(textchunk, result, idx, opts, logger)
    return result


def parallel_worker(callback, file, opts, logger, num_of_processes=8, worker=_worker):
    filesize = os.path.getsize(file)
    logger.info("File size %d" % filesize)
    offsets = find_textfile_split_points(file, num_of_processes)
    # append end of file
    offsets.append(filesize)
    futures = [None] * num_of_processes
    with ThreadPoolExecutor(max_workers=num_of_processes) as p:
        for idx in list(range(num_of_processes)):
            start_offset = offsets[idx]
            end_offset = offsets[idx + 1]
            futures[idx] = p.submit(worker, callback, file, idx, start_offset, end_offset, opts, logger)
    return futures
