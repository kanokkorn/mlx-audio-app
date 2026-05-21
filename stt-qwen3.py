import os
import sys
from mlx_audio.stt import load


def dirwalk(dirpath):
    """Yields paths to all mp3 files found in the directory tree."""
    for root, _, files in os.walk(dirpath):
        for file in files:
            if file.lower().endswith(".mp3"):
                yield os.path.join(root, file)


def speech2text(model, filepath):
    """Transcribes a single file using the pre-loaded model."""
    print(f"Processing: {filepath}...")
    try:
        # Note: Ensure your version of mlx_audio supports the language
        result = model.generate(filepath, language="Thai")
        with open("result.txt", "a", encoding="utf-8") as f:
            f.write(f"--- {os.path.basename(filepath)} ---\n")
            f.write(result.text + "\n\n")
        print(f"Done: {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python stt-qwen3.py <directory_path>")
        sys.exit(1)

    target_dir = sys.argv[1]
    if not os.path.isdir(target_dir):
        print(f"Error: {target_dir} is not a valid directory.")
        sys.exit(1)

    # Load model ONCE outside the loop for efficiency
    print("Loading model...")
    # Ensure the model path is correct
    model = load("mlx-community/Qwen3-ASR-0.6B-8bit")
    # model = load("facebook/mms-1b-fl102")

    for mp3_file in dirwalk(target_dir):
        speech2text(model, mp3_file)

    print("\nAll tasks complete. Results saved to result.txt")
