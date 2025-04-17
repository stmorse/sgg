import json
import os

import bz2                # for .bz2
import lzma               # for .xz
import zstandard as zstd  # for .zst
import zarr

import numpy as np

DATA_PATH = '/sciclone/data10/twford/reddit/reddit/comments/'
EMBED_PATH = '/sciclone/geograd/stmorse/reddit/embeddings'

def iterate_months(s_year, s_month, e_year, e_month):
    year, month = s_year, s_month
    while (year < e_year) or (year == e_year and month <= e_month):
        yield year, month
        month += 1
        if month > 12:
            month = 1
            year += 1

def open_compressed(file_path):
    if file_path.endswith('.bz2'):
        return bz2.BZ2File(file_path, 'rb')
    elif file_path.endswith('.xz'):
        return lzma.open(file_path, 'rb')
    elif file_path.endswith('.zst'):
        # For .zst, return a stream reader
        f = open(file_path, 'rb')  # Open file in binary mode
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        return dctx.stream_reader(f)
        # return dctx.read_to_iter(f)
    else:
        raise ValueError('Unsupported file extension.')


def parse_line(line, return_type='metadata', metadata=['id'], 
               filter=True, truncate=0):
    entry = json.loads(line.decode('utf-8'))

    if filter:
        if 'body' not in entry or entry['author'] == '[deleted]':
            return None
    
    if return_type == 'metadata':
        res = {k: entry[k] for k in metadata}
        return res
    elif return_type == 'sentences':
        body = entry['body']
        if truncate > 0:
            body = body[:truncate]
        return body
    else: 
        print('Error: return_type not recognized. Must be metadata or sentences.')

    return None

def read_file(
        year, month, 
        return_type='metadata', 
        metadata=['id'],
        chunk_size=10000,
        data_path=DATA_PATH):
    """
    Read JSON entries from a compressed file, extract fields,
    and yield chunks of size `chunk_size`.

    return_type : 'metadata' or 'sentences'
    """

    # Find the file with the given year and month
    file_name = next(
        (f for f in os.listdir(data_path) if f.startswith(f'RC_{year}-{month:02}')),
        None
    )

    if not file_name:
        raise FileNotFoundError(f"No file found for RC_{year}-{month:02} in {data_path}")

    # Extract the file extension
    ext = os.path.splitext(file_name)[1]

    filepath = os.path.join(data_path, file_name)
    print('debug', filepath)

    buffer = []         # To store 'body' fields in chunks
    byte_buffer = b""   # For handling partial lines in `.zst` files

    with open_compressed(filepath) as f:
        if ext == 'bz2':
            for line in f:
                res = parse_line(
                    line, 
                    return_type=return_type,
                    metadata=metadata,
                    filter=True,
                    truncate=2000
                    )

                if res is None:
                    continue

                # Add to the chunk buffer
                buffer.append(res)
                if len(buffer) >= chunk_size:
                    yield buffer
                buffer = []
        else:
            # Iterate over the file
            for chunk in iter(lambda: f.read(8192), b""):  # Read file in binary chunks
                byte_buffer += chunk

                # Process each line in the byte buffer
                while b"\n" in byte_buffer:
                    line, byte_buffer = byte_buffer.split(b"\n", 1)

                    res = parse_line(
                        line, 
                        return_type=return_type,
                        metadata=metadata,
                        filter=True,
                        truncate=2000
                    )

                    if res is None:
                        continue

                    # Add to the chunk buffer
                    buffer.append(res)
                    if len(buffer) >= chunk_size:
                        yield buffer
                        buffer = []

            # Handle any remaining partial JSON line
            if byte_buffer:
                res = parse_line(
                    line, 
                    return_type=return_type,
                    metadata=metadata,
                    filter=True,
                    truncate=2000
                )
                buffer.append(res)

        # Yield any leftovers in the chunk buffer
        if buffer:
            yield buffer

def load_embeddings(year, month, embed_path=EMBED_PATH):
    file_path = os.path.join(
        embed_path,
        f'embeddings_{year}-{month:02}.npz'
    )
    with open(file_path, 'rb') as f:
        embeddings = np.load(f)['embeddings']
        return embeddings