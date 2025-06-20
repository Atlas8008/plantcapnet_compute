import io
import time
import errno

from multiprocessing import Manager
from PIL import Image


MAX_RETRIES = 30


class ImageServer:
    """A class for handling image loading and caching."""
    def __init__(self, cache_files=False):
        self.cache_files = cache_files
        self.cache = Manager().dict()

    def get(self, path, mode="RGB"):
        if self.cache_files:
            if path not in self.cache:
                self.cache[path] = self._read_from_file(path)

            img = Image.open(io.BytesIO(self.cache[path]))
        else:
            img = Image.open(io.BytesIO(self._read_from_file(path)))
            #img = Image.open(path)

        return img.convert(mode)

    def _read_from_file(self, path):
        data_loaded = False
        retries = 0

        while not data_loaded:
            try:
                with open(path, "rb") as f:
                    data = f.read()

                data_loaded = True
            except OSError as e:
                if e.errno == errno.EPIPE or e.errno == errno.ECONNRESET:
                    raise e

                print("Encountered OSError while loading: {}\nTrying again...".format(e))

                retries += 1

                if retries >= MAX_RETRIES:
                    raise e

                time.sleep(5)

        return data


    def __getitem__(self, path):
        return self.get(path)
