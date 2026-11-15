"""Write the generated test image to a file."""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pickplace.synthetic import render_scene  # noqa: E402

if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/synthetic_photo.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), render_scene())
    print(f"wrote {out}")
