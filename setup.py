import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='PTI',
    version='0.2.4',
    author='I.N.Tzortzis',
    author_email='i.n.tzortzis@gmail.com',
    description='PyTorch Training Inference Framework',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/itzortzis/pytorch-training-inference',
    project_urls = {
        "Code": "https://github.com/itzortzis/pytorch-training-inference",
        "Bug Tracker": "https://github.com/itzortzis/pytorch-training-inference/issues"
    },
    license='GPL-3.0',
    packages=['pti'],
    install_requires=['numpy', 'torch', 'tqdm', 'matplotlib']#, 'csv', 'sklearn.cluster', 'skimage.color', 'calendar', 'torchmetrics'],
)