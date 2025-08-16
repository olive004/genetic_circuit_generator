

# Test that naming conventions in DataWriter are as expected.

""" Expectations:

Output:
- Root experiment run dir
    - output_summary.csv file tracking all output files
    - experiment.json
        - config file used

- Per circuit 
    - Subfolders: files for current circuit permutation go into a subfolder accessed by 'write_dir'
        - header / run info
            - mutations.csv
        - data
            - report.json
            - *_data.csv
        - visualisations
            - graph.html
            - *.png
"""

## 1. Write results into a list of subfolders

## 2. Write results into original write directory

## 3. If multiple are nested
