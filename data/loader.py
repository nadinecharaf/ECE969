from datasets import Dataset
import config


def load_split(split: str = "test") -> Dataset:
    """Load a single ScienceQA split from local Arrow files.

    Args:
        split: One of "train", "validation", "test".

    Returns:
        A HuggingFace Dataset object.
    """
    arrow_file = config.DATASET_PATH / f"science_qa-{split}.arrow"
    if not arrow_file.exists():
        raise FileNotFoundError(f"Arrow file not found: {arrow_file}")
    return Dataset.from_file(str(arrow_file))


def load_all_splits() -> dict[str, Dataset]:
    """Load train / validation / test splits."""
    return {split: load_split(split) for split in ("train", "validation", "test")}
