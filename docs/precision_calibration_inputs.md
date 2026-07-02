# Precision calibration inputs

The detector can continue operating with the checked-in estimated calibration.
The following user inputs are required before validating millimeter accuracy:

1. Exact playing-surface length in millimeters, measured cushion nose to
   opposite cushion nose.
2. Exact playing-surface width in millimeters, measured cushion nose to
   opposite cushion nose.
3. A four-click calibration using a full-resolution empty-table image:

   ```powershell
   python tools/click_table_corners.py `
     --image Media/01_empty_table/DSC00543.JPG
   ```

   Click top-left, top-right, bottom-right, bottom-left. At pocket openings,
   click the virtual intersection of the straight cushion-nose lines.
4. Confirmation of the actual ball diameter. The current configuration uses
   52.5 mm.

For measured coordinate error rather than visual inspection, the next dataset
should include several balls placed at independently measured X/Y locations.
Five to ten positions distributed across the center, corners, and cushion
edges are sufficient for the first error map.

Lens calibration should follow after this manual-homography baseline is
measured. It is required before claiming a few-millimeter full-table accuracy.

