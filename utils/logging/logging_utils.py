import os
from datetime import datetime


def log_to_file(fname, contents, final_spacing=True, timestamp=True):
    folder = os.path.split(fname)[0]

    if folder:
        os.makedirs(folder, exist_ok=True)

    with open(fname, "a") as f:
        if isinstance(contents, tuple):
            contents = list(contents)

        if not isinstance(contents, list):
            contents = [contents]

        if timestamp:
            contents = [str(datetime.now()) + ": "] + contents

        for content in contents:
            f.write(str(content) + "\n")

        if final_spacing:
            f.write("\n")