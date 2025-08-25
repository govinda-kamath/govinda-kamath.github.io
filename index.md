---
layout: single
author_profile: true
comments: false
title: false
---

### About Me


I am a research scientist at [10x Genomics](https://www.10xgenomics.com/) in California.
My research journey includes a postdoctoral fellowship at 
the [Microsoft Research New England Lab](https://www.microsoft.com/en-us/research/lab/microsoft-research-new-england/)
with [Lester Mackey](https://www.microsoft.com/en-us/research/people/lmackey/),
a PhD from Stanford's [Information Systems Lab](http://www-isl.stanford.edu/index.html), 
where I was advised by Professor [David Tse](https://web.stanford.edu/~dntse/), 
and a masters in [Electrical Communication Engineering](http://ece.iisc.ernet.in) from Indian Institute of science
with Professor [P Vijay Kumar](http://ece.iisc.ernet.in/~vijay/).
My work has consistently had the theme of designing and 
deploying scalable machine learning algorithms to solve real problems,
and showing their goodness, with problems often coming from computational genomics.


### Current Work

At 10x Genomics, I work on developing algorithms and machine learning
to efficiently process large amounts of data generated from single-cell-RNA-seq and 
spatial-transcriptomics in a principled manner. Single-cell-RNA-seq and spatial-transcriptomics
present statistical challenges because experiment design is in the setting is hard;
and the data generated is mostly observational. In addition, the scale of the data
presents algorithmic challenges. 

I work in a
with people with diverse backgrounds in a team developing the products. 
In particular I worked on the [Visium HD product](https://www.10xgenomics.com/support/spatial-gene-expression-hd),
where we detect genes on different spatial areas of a tissue slice. 
I was involved with most aspects of the product
development; from interfacing with other teams to model what we were trying to build,
to figuring out metrics we had to optimise for a good product to developing
algorithms to process our data. The data 
we generate can be thought of as an image with around 20,000 channels -- each corresponding
to a gene; and each being sparse in addition to a H&E image.
Processing this presents us with many interesting algorithmic and statistical problems.

I have also worked in and led teams that developed and shipped machine learning models 
from end to end -- most notably in the context of 
[annotating cell-types in singe cell data](https://www.10xgenomics.com/support/software/cell-ranger/latest/algorithms-overview/cr-cell-annotation-algorithm#mouse-models); and 
in [annotating nuclei from microscope images stained with H&E](https://www.10xgenomics.com/support/software/space-ranger/latest/algorithms-overview/image-processing#cell-segmentation).
In both those cases, we worked on end-to-end development of the product; from
figuring out what data we would need; to setting up data collection,
cleaning and labelling; to building, packaging and shipping the model.

Working in the industry got me writing code and thus in programming languages. In particular, I enjoy
coding in [Rust](https://www.rust-lang.org/).