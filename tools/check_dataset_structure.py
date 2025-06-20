import os
import json
import csv
import os
import json
import csv

def check_dataset_structure(dataset_path):
    required_files = ["data.json", "class_names.txt"]
    required_dirs = ["annotations", "images", "splits"]
    optional_dirs = ["masks"]

    # Check if required files exist
    for file in required_files:
        if not os.path.exists(os.path.join(dataset_path, file)):
            print(f"Missing required file: {file}")
            return False

    # Check if required directories exist
    for directory in required_dirs:
        if not os.path.exists(os.path.join(dataset_path, directory)):
            print(f"Missing required directory: {directory}")
            return False

    # Check if optional directories exist (optional check, can be ignored)
    for directory in optional_dirs:
        if not os.path.exists(os.path.join(dataset_path, directory)):
            print(f"Optional directory missing: {directory}")

    # Validate data.json structure
    data_json_path = os.path.join(dataset_path, "data.json")
    try:
        with open(data_json_path, "r") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                print("data.json should be a dictionary.")
                return False

            for mode, sites in data.items():
                if not isinstance(sites, dict):
                    print(f"Invalid structure under mode {mode} in data.json")
                    return False
                for site, images in sites.items():
                    if not isinstance(images, list):
                        print(f"Invalid image list under site {site} in mode {mode}")
                        return False
                    for entry in images:
                        if "image" not in entry or "index" not in entry:
                            print(f"Missing required keys in an entry under site {site}, mode {mode}")
                            return False
    except Exception as e:
        print(f"Error reading data.json: {e}")
        return False

    # Validate annotations structure
    annotation_dir = os.path.join(dataset_path, "annotations")
    for file in os.listdir(annotation_dir):
        if file.endswith(".csv"):
            file_path = os.path.join(annotation_dir, file)
            try:
                with open(file_path, "r") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    required_columns = ["plot", "year", "month", "day", "hour", "minute"]
                    if not all(col in header for col in required_columns):
                        print(f"Invalid annotation structure in {file}")
                        return False
            except Exception as e:
                print(f"Error reading annotation file {file}: {e}")
                return False

    # Validate splits
    splits_dir = os.path.join(dataset_path, "splits")
    required_splits = ["split_trainval.json"]
    for split in required_splits:
        if not os.path.exists(os.path.join(splits_dir, split)):
            print(f"Missing required split file: {split}")
            return False

    print("Dataset structure is valid.")
    return True


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python check_dataset_structure.py <path_to_dataset>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    check_dataset_structure(dataset_path)
