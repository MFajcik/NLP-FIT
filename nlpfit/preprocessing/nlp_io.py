# PyVersion: 3.6
# Authors: Martin Fajčík FIT@BUT
import os


def read_word_lists(*args, **kwargs):
    chunk = read_word_chunks(*args, **kwargs)
    report_bytesread = kwargs.get("report_bytesread", False)
    for item in chunk:
        yield (item[0].split(),item[1]) if report_bytesread else item.split()


def read_word_chunks(filename: str, bytes_to_read: int,
                     startoffset: int = -1, endoffset: int = -1, report_bytesread = False):
    """
    Lazy file reading,
    :param startoffset: file offset, if -1 it is set to start of file
    :param endoffset: file offset, if -1 it is set to end of file
    :param filename: file to read
    :param bytes_to_read:
    :return generator that reads bytes_to_read bytes each call
            and yields properly aligned chunks of text
    """
    last = b""
    if startoffset < 0:
        startoffset = 0
    if endoffset < 0:
        endoffset = os.path.getsize(filename)

    with open(filename, mode="rb") as inp:
        inp.seek(startoffset)
        bytes_read = 0
        offset = inp.tell()
        while offset < endoffset:
            if (offset + bytes_to_read > endoffset):
                bytes_to_read = endoffset - offset
            buf = inp.read(bytes_to_read)
            bytes_read= inp.tell()
            offset = inp.tell()
            if not buf:
                break
            # print("Last:%s|BUF:%s"%(last,buf))
            buf = last + buf
            # Last one can be incomplete, we will pop it and use it next time!
            # 19.03 fixed bug related to rsplit, trailing space was destroyed b4
            # In: "tak teda aha ".rsplit(None, 1)
            # Out: ['tak teda', 'aha']

            buf, last = buf.rsplit(None, 1) if not str.isspace(chr(buf[-1])) else (buf, b"")
            # print("Last==%s"%last)
            yield (buf.decode("utf-8"),bytes_read) if report_bytesread else buf.decode("utf-8")
        yield ((b' ' + last).decode("utf-8"),bytes_read) if report_bytesread else (b' ' + last).decode("utf-8")


def read_words(filename, chars_to_read):
    """
    Lazy file reading,
    :param filename: file to read
    :param chars_to_read:
    :return generator that reads bytes_to_read bytes each call
            and returns word by word
    """
    last = ""
    with open(filename) as inp:
        while True:
            buf = inp.read(chars_to_read)
            if not buf:
                break
            words = (last + buf).split()
            # Last one can be incomplete, we will pop it and use it next time!
            last = words.pop()
            for word in words:
                yield word
        yield last
