import setuptools

with open("README.md", "r") as f:
    desc = f.read()

setuptools.setup(
    name="fast-prototype",
    version="0.1.1",
    author="Vishal Keshav",
    author_email="vishal.keshav.1993@gmail.com",
    description="Fast way to prototype research ideas",
    long_description = desc,
    long_description_content_type="text/markdown",
    url="https://github.com/vishal-keshav/fast_prototype",
    packages=setuptools.find_packages(),
    #install_requires=['']
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
)
