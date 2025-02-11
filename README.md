# troutpy

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/sergiomarco25/troutpy/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/troutpy

Package for the analysis of transcripts outside segmented cells in python


![Alt text][images/logo_fish.png]

## Getting started

Please refer to the [documentation][],
in particular, the [API documentation][].

## Installation

You neeed to have Python 3.10 or newer installed on your system.
If you don't have Python installed, we recommend installing [Mambaforge][].

There are several alternative options to install troutpy:

<!--
1) Install the latest release of `troutpy` from [PyPI][]:

```bash
pip install troutpy
```
-->

1. Install the latest development version:

```bash
pip install git+https://github.com/theislab/troutpy.git@main
```


## Usage
Please have a look at the [Usage documentation](https://troutpy.readthedocs.io/en/latest/) and the [tutorials](https://troutpy.readthedocs.io/en/latest/).

```python
import troutpy as tp
```

## Release notes

See the [changelog][].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

## Citation

> t.b.a

[mambaforge]: https://github.com/conda-forge/miniforge#mambaforge
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/theislab/troutpy/issues
[tests]: https://github.com/theislab/troutpy/actions/workflows/test.yml
[documentation]: https://troutpy.readthedocs.io
[changelog]: https://troutpy.readthedocs.io/en/latest/changelog.html
[api documentation]: https://troutpy.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/troutpy
[images/logo_fish.png]: images/logo_fish.png
