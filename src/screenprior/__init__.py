"""screenprior -- batch active learning for CRISPR screens with a corpus of completed screens.

The corpus is the point. Every method in this line fits a surrogate on the genes acquired so far in
the screen being run; BioGRID ORCS publishes over a thousand completed genome-scale human screens
with their authors' own hit calls, and that corpus is a prior over which screen this one resembles.
"""

from screenprior import comparators, corpus, features, harness, metrics, policies, surrogate

__version__ = "0.1.0"
__all__ = ["comparators", "corpus", "features", "harness", "metrics", "policies",
           "surrogate"]
