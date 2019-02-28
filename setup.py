from setuptools import setup, find_packages


setup(
    name='tadataka',
    description='A high-level Python Visual SLAM package',
    url='http://github.com/IshitaTakeshi/Tadataka',
    author='Takeshi Ishita',
    author_email='ishitah.takeshi@gmail.com',
    license='MIT',
    packages=['tadataka'],
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'scikit-image',
        'tqdm'
    ]
)
