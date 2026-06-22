import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm


MAPPING_CSV = "split_mapping.csv"
OUTPUT_DIR = "data"
ROOT_DIR = r"C:\Users\levie\Downloads\archive\S3"


def main() -> None:
	csv_path = Path(MAPPING_CSV)
	if not csv_path.exists():
		raise FileNotFoundError(f"CSV not found: {csv_path}")

	df = pd.read_csv(csv_path)
	required = {"original_path", "final_path"}
	missing = required - set(df.columns)
	if missing:
		raise ValueError(f"CSV missing columns: {', '.join(sorted(missing))}")

	output_dir = Path(OUTPUT_DIR)
	output_dir.mkdir(parents=True, exist_ok=True)

	missing_sources = 0
	progress = tqdm(total=len(df), desc="Write data", unit="img")
	for _, row in df.iterrows():
		src = Path(row["original_path"])
		if not src.is_absolute():
			src = Path(ROOT_DIR) / src
		final_path = Path(row["final_path"])
		if final_path.is_absolute():
			dst = final_path
		else:
			dst = output_dir / final_path

		if not src.exists():
			missing_sources += 1
			progress.update(1)
			continue

		dst.parent.mkdir(parents=True, exist_ok=True)
		shutil.copy2(str(src), str(dst))
		progress.update(1)

	progress.close()
	if missing_sources:
		print(f"Skipped missing source files: {missing_sources}")


if __name__ == "__main__":
	main()
