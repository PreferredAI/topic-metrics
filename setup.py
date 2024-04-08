from setuptools import setup, find_packages
from topic_metrics import __version__

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = ['numpy', 'tqdm']

setup(
    author="JP Lim",
    author_email='jiapeng.lim.2021@phdcs.smu.edu.sg',
    python_requires='>=3.6',
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
    ],
    description="topic-metrics: preferred tool for meddling with topics",
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords='topic_metrics',
    name='topic_metrics',
    packages=find_packages(include=['topic_metrics', 'topic_metrics.*']),
    url='https://github.com/PreferredAI/topic-metrics',
    version=__version__
)