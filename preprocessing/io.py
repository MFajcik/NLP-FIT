# PyVersion: 3.6
# Authors: Martin Fajčík FIT@BUT

def read_word_chunks_with_offset(filename: str, bytes_to_read: int,
                                startoffset: int, endoffset: int):
    """
    Lazy file reading,
    :param filename: file to read
    :param bytes_to_read:
    :return generator that reads bytes_to_read bytes each call
            and yields properly aligned chunks of text
    """
    last = ""
    with open(filename, mode="rb") as inp:
        inp.seek(startoffset)
        offset = inp.tell()
        while offset<endoffset:
            if (offset+bytes_to_read>endoffset):
                bytes_to_read = endoffset-offset
            buf = str(inp.read(bytes_to_read),encoding='utf-8',errors="ignore")
            offset = inp.tell()
            if not buf:
                break
            buf = last + buf
            # Last one can be incomplete, we will pop it and use it next time!
            buf,last = buf.rsplit(None, 1)
            yield buf
        yield ' '+last

def read_word_lists_with_offset(filename: str, bytes_to_read: int,
                                startoffset: int, endoffset: int):
    """
    Lazy file reading,
    :param filename: file to read
    :param bytes_to_read:
    :return generator that reads bytes_to_read bytes each call
            and returns list of words
    """
    last = ""
    with open(filename, mode="rb") as inp:
        inp.seek(startoffset)
        offset = inp.tell()
        while offset<endoffset:
            if (offset+bytes_to_read>endoffset):
                bytes_to_read = endoffset-offset
            buf = str(inp.read(bytes_to_read),encoding='utf-8',errors="ignore")
            offset = inp.tell()
            if not buf:
                break
            words = (last + buf).split()
            # Last one can be incomplete, we will pop it and use it next time!
            last = words.pop()
            yield words
        yield [last]

def read_word_lists(filename: str, chars_to_read: int):
    """
    Lazy file reading,
    :param filename: file to read
    :param chars_to_read:
    :return generator that reads bytes_to_read bytes each call
            and returns list of words
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
            yield words
        yield [last]


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
