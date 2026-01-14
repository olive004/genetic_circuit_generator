

import os

def add_license_header(file_path, license_text):
    # Read the contents of the file
    with open(file_path, 'r') as file:
        content = file.read()

    # Add the license header at the beginning of the file
    with open(file_path, 'w') as file:
        file.write(license_text + '\n' + content)

def add_license_to_files(directory, license_text):
    # Iterate over all files in the directory
    for root, _, files in os.walk(directory):
        for file in files:
            # Check if the file is a source file (e.g., .py, .java, .cpp)
            if file.endswith(('.py')) and (not file.startswith('__init__')):
                file_path = os.path.join(root, file)
                add_license_header(file_path, license_text)
                print(f"License added to: {file_path}")
            # if file.startswith('__init__') and file.endswith(('.py')):
            #     file_path = os.path.join(root, file)
            #     print(file_path)
            #     with open(file_path, 'w') as file:
            #         file.write('')
                

# Example usage
if __name__ == '__main__':
    directory_path = 'synbio_morpher'
    license_header = '''
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    '''

    add_license_to_files(directory_path, license_header)