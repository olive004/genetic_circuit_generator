from setuptools import setup, find_packages

if __name__ == "__main__":
    # setup()

    setup(
        name='synbio_morpher',
        authors=[
            {'name': "Olivia Gallup", 'email': "olivia.gallup@gmail.com"}
        ],
        version='0.0.903',
        license='LICENSE',
        description='Build, simulate, and analyse genetic circuits and the effect of mutations.',
        readme="README.md",
        packages=find_packages(exclude=['__pycache__']),
        package_data={'': ['*.txt', '*.json', '*.md', '*.csv', '*.sh', '*.fasta']},
        include_package_data=True,
        install_requires=[],
    )
