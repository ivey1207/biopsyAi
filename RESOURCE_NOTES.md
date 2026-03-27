# Resource-driven improvements

This project was updated using ideas from mask-focused resources:

- Binary mask cleanup utilities and component filtering patterns.
- LIDC-style binary-mask generation mindset (strict binary output, matching names/sizes).
- Ideal mask pipeline concept (probability -> threshold -> binary mask).
- Segmentation visualization style (binary mask + contour + distance map).

Applied upgrades in this codebase:

1. `models/segmentation/segment.py`
   - Added `--threshold-mode` (`fixed`, `otsu`, `percentile`).
   - Added `--percentile` for robust adaptive thresholding.
   - Added `--max-components` to keep top connected components.
   - Improved cleanup with morphology + connected-component area filtering.
   - Supports `jpg/png/jpeg` test images.

2. `app/streamlit_app.py`
   - Added threshold mode selector and percentile control.
   - Added max connected components control.
   - Added distance-map visualization in addition to binary mask/contour/probability map.
   - Shows final threshold used and coverage percentage.

3. `README.md`
   - Added advanced segmentation inference examples with adaptive thresholding.
