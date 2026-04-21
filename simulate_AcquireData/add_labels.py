#!/usr/bin/env python3
"""
add_labels.py
Add ground truth labels to the deck simulation CSV for training.

Input:  w48_v40.csv  (TIME, N1_UZ, N1_AZ, N7_UZ, N7_AZ)
Output: w48_v40_labeled.csv  (same + 4 label columns)

Label columns:
  front_wheel_pos  -- front wheel distance from left end of deck (m)
  rear_wheel_pos   -- rear wheel distance from left end of deck (m)
  front_axle_wt   -- front axle weight (N), 0 when off deck
  rear_axle_wt    -- rear axle weight (N), 0 when off deck

Logic:
  front_x = speed * time
  rear_x  = front_x - axle_dist
  If wheel x is outside [0, 40]: position = 0, weight = 0
"""

import csv
import os

# ============================================================
# Parameters -- modify these to match the simulation case
# ============================================================
AXLE_WEIGHT = 45000.0   # axle weight in Newtons (e.g. 48000 N = 48 kN)
SPEED       = 30.0       # vehicle speed (m/s)
AXLE_DIST   = 8.0        # wheelbase (m), distance front-to-rear axle
DECK_LENGTH = 40.0       # deck length (m)

INPUT_FILE  = "w45_v30.csv"
OUTPUT_FILE = "w45_v30_labeled.csv"
# ============================================================

def compute_labels(t):
    """
    Given time t (seconds), return (front_pos, rear_pos, front_wt, rear_wt).
    When wheel is off deck (x < 0 or x > 40), position and weight are 0.
    """
    front_x = SPEED * t
    rear_x  = front_x - AXLE_DIST

    # Front wheel
    if 0.0 <= front_x <= DECK_LENGTH:
        front_pos = front_x
        front_wt  = AXLE_WEIGHT
    else:
        front_pos = 0.0
        front_wt  = 0.0

    # Rear wheel
    if 0.0 <= rear_x <= DECK_LENGTH:
        rear_pos = rear_x
        rear_wt  = AXLE_WEIGHT
    else:
        rear_pos = 0.0
        rear_wt  = 0.0

    return front_pos, rear_pos, front_wt, rear_wt


def main():
    input_path  = os.path.join(os.path.dirname(__file__), INPUT_FILE)
    output_path = os.path.join(os.path.dirname(__file__), OUTPUT_FILE)

    if not os.path.exists(input_path):
        print(f"Error: input file not found: {input_path}")
        return

    print(f"Reading:  {input_path}")
    print(f"Writing:  {output_path}")
    print(f"Params:   weight={AXLE_WEIGHT}N, speed={SPEED}m/s, axle_dist={AXLE_DIST}m")

    with open(input_path, "r", newline="", encoding="utf-8") as fin, \
         open(output_path, "w", newline="", encoding="utf-8") as fout:

        reader = csv.reader(fin)
        writer = csv.writer(fout)

        # Read and write header
        header = next(reader)
        new_header = header + ["front_wheel_pos", "rear_wheel_pos",
                               "front_axle_wt", "rear_axle_wt"]
        writer.writerow(new_header)
        print(f"Columns:  {new_header}")

        # Process each data row
        row_count = 0
        for row in reader:
            if len(row) < 1 or row[0].strip() == "":
                continue

            t = float(row[0])
            front_pos, rear_pos, front_wt, rear_wt = compute_labels(t)

            new_row = row + [front_pos, rear_pos, front_wt, rear_wt]
            writer.writerow(new_row)
            row_count += 1

        print(f"Rows written: {row_count}")

    print("Done.")


if __name__ == "__main__":
    main()
