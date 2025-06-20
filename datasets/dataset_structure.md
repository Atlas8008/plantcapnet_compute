## Dataset Specification

This dataset follows a structured format for organizing images, masks (optional), annotations, and splits. It is designed to support various annotation modes, where each mode represents a different way of labeling the data (e.g., daily or weekly labels). The `index` field in `data.json` links each image entry to a corresponding annotation row in the annotation files. Note that image files can also have a different format than jpg (e.g., tif or png). For a full list of supported format, please refer to the documentation of the [pillow package](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html).

```
<Dataset Folder>/
├── data.json  # JSON file containing metadata and indexing for images and annotations
├── class_names.txt  # Text file containing the species/class names for all classes in the dataset, one per line
├── annotations  # CSV files with labeled data
│   ├── cover_<Mode 1>.csv  # Cover-related annotations for Mode 1 (e.g., daily, weekly)
│   ├── flowering_<Mode 1>.csv  # Flowering-related annotations for Mode 1
│   ├── senescence_<Mode 1>.csv  # Senescence-related annotations for Mode 1
│   ├── cover_<Mode 2>.csv  # Cover-related annotations for Mode 2
│   ├── flowering_<Mode 2>.csv  # Flowering-related annotations for Mode 2
│   ├── senescence_<Mode 2>.csv  # Senescence-related annotations for Mode 2
│   ├── ...  # Additional modes as needed
├── images  # Image files grouped by site
│   ├── <Site 1>
│   │   ├── <Site 1, Image 1>.jpg  # Image file for Site 1
│   │   ├── <Site 1, Image 2>.jpg
│   │   ├── ...
│   ├── <Site 2>
│   │   ├── <Site 2, Image 1>.jpg  # Image file for Site 2
│   │   ├── <Site 2, Image 2>.jpg
│   │   ├── ...
│   ├── ...  # Additional sites
├── masks  # Optional segmentation masks, structured similarly to images
│   ├── <Site 1>
│   │   ├── <Site 1, Mask image 1>.jpg  # Mask for corresponding image
│   │   ├── <Site 1, Mask image 2>.jpg
│   │   ├── ...
│   ├── <Site 2>
│   │   ├── <Site 2, Mask image 1>.jpg
│   │   ├── <Site 2, Mask image 2>.jpg
│   │   ├── ...
│   ├── ...  # Additional sites
└── splits  # JSON files defining dataset splits for training/validation
    └── split_trainval.json  # Training/Validation split
```

### Structure of `data.json`

The `data.json` file maps images to their corresponding annotations. It contains multiple modes, where each mode represents a user-defined labeling approach (e.g., daily, weekly). Each image entry includes an optional mask and an `index` that links to a specific annotation row.

```
{
 "<Mode 1>": {
  "<Site 1>": [
   {
    "image": "<Site 1>/<Site 1, Image 1>.jpg",  # Path to image
    "mask": "<Site 1>/<Site 1, Mask image 1>.png",  # Optional mask path
    "index": {  # Links to an annotation row in corresponding CSV file
     "site": "<Site 1>",
     "year": <Year>,
     "month": <Month>,
     "day": <Day>,
     "hour": <Hour>,
     "minute": <Minute>
    }
   },
   {
    "image": "<Site 1>/<Site 1, Image 2>.jpg",
    "mask": "<Site 1>/<Site 1, Mask image 2>.png",
    "index": {
     "site": "<Site 1>",
     "year": <Year>,
     "month": <Month>,
     "day": <Day>,
     "hour": <Hour>,
     "minute": <Minute>
    }
   },
   ...
  ],
  "<Site 2>": [
   ...
  ],
  ...
 },
 "<Mode 2>": {
    ...
 },
 ...
}
```

### Structure of Split File(s) (e.g., `split_trainval.json`)

A split file contains a key "train" mapping to a list of sites to train on, and a key "validation" mapping to the sites to be used for validation.

```
{
 "train": [
  "<Site 1>",
  "<Site 2>",
  ...
 ],
 "validation": [
  "<Site 3>",
  "<Site 4>",
  ...
 ]
}
```

### Structure of Annotation Files (cover/flowering/senescence)

Each annotation file corresponds to a specific mode (e.g., daily or weekly) and contains labeled data. The `index` field in `data.json` points to a unique row in these CSV files, ensuring correct alignment between images and labels.

| site     | year  | month | day | hour | minute | <Species 1> | <Species 2> | ... |
| ---------|-------|-------|-----|------|--------|-------------|-------------|-----|
| <Site 1> | 2024  | 12    | 03  | 12   | 00     | 30          | 50          | ... |
| <Site 1> | 2024  | 12    | 03  | 13   | 00     | 35          | 55          | ... |
| ...      | ...   | ...   | ... | ...  | ...    | ...         | ...         | ... |
| <Site 2> | 2024  | 12    | 03  | 13   | 00     | 30          | 50          | ... |
| ...      | ...   | ...   | ... | ...  | ...    | ...         | ...         | ... |

Each row contains:
- `site`: The site where the image was captured.
- `year`, `month`, `day`, `hour`, `minute`: Timestamp of the annotation.
- `<Species>`: Numerical values representing the annotated cover, flowering, or senescence for each species.

The dataset structure allows flexible annotation modes, making it adaptable to different research needs.
