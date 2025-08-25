---
layout: single
author_profile: true
comments: false
title: Past Research
---

The following are a sample of the problems I have worked on 
during my doctoral and post-doctoral studies (prior to joining 10x Genomics):

- **Genome assembly**: The genome of an organism is a long strings of A,C,T,G. 
Current sequencing technology allows us to obtain noisy substrings of the genome called reads from the genome. 
These noisy reads are of length around 10,000 while the genome is of length in millions or billions.
Assembling the genome from reads is a fundamental problem of computational genomics. 
I worked on this from both a theoretic and a practical perspective.<br />
Genome assembly is constrained by repeats in the genome. We realised that the output of an assembler should 
be a graph where every sequence corresponding to a genome is an eulerian cycle 
(or an eulerian tour if the genome is linear). We wrote a practical assembler called [HINGE](https://github.com/HingeAssembler/HINGE)
which was published in [Genome Research](http://genome.cshlp.org/content/27/5/747), 
and is still under active development with quite a few users primarily doing bacterial genomics. The current focus here is reducing the memory footprint. The theoretic results, which showed goodness of simplified version of HINGE from a rate distortion perspective
were presented at 
[International Symposium of Information Theory (ISIT)](https://arxiv.org/abs/1605.01941).

- **Using Monte-Carlo tree search to speed up common ML algorithms**:
We consider common machine learning algorithms such as k-Nearest Neighbours, k-means and show that a monte-carlo search based algorithm can 
provide speedups of the order of 10x in practice. We also prove theoretical results using the multi-armed bandit framework. 
This was presented in part in NIPS-2017 workshop, in [AISTATS-2018](https://arxiv.org/abs/1711.00817) and is
[IEEE Journal on Selected Areas in Information Theory](https://arxiv.org/abs/1805.08321v2).


- **Low-Rank models for sequence alignment**:  We considered a class of algorithms for sequence alignment - seed 
and extend algorithms, and drew connections to low rank models. We derived spectral 
estimators for such models and then used the framework we developed in 2. to speed up its computation. 
This was published in [Recomb 2020](https://link.springer.com/chapter/10.1007/978-3-030-45257-5_15),
[Cell Patterns](https://www.cell.com/patterns/pdf/S2666-3899(20)30109-4.pdf)
and [NeurIPS 2020](https://papers.nips.cc/paper/2020/file/54e0e46b6647aa736c13ef9d09eab432-Paper.pdf)

- **Knowledge Distillation And Semiparametric Inference**: 
We studied the problem of knowledge distillation - where it has been observed
that training a small model using logits soft class labels  generated using a large 
model trained from the data does better than directly training the small model using the data.  
We cast knowledge distillation as a semiparametric inference problem and derive several new guarantees 
for the prediction error of standard distillation. This will be published in [ICLR 2021](https://openreview.net/forum?id=m4UCf24r0Y).


- **Single Cell RNA-seq**: RNA is an intermediate compound often created as an intermediate between DNA and protein.
RNA transcripts are much smaller than DNA (of length tens of thousand instead of billions), and the number of 
molecule of each RNA transcript is dynamically varying with time (and the type of cell in the organism), while
all the cells in any organism has the same DNA. 
RNA-seq is a sequencing assay that sequences RNA rather than DNA. 
With advances in the bio-chemistry, researchers have been able to sequence RNA from single cells. 
This presents interesting computational problems, as there is very little RNA in each cell. 
The fact that millions of cells are sequenced and each cell has an expression (feature vector) of tens
of thousands of dimensions makes the problem more challenging.<br />
One basic application of this technology is to be able to cluster cells from their RNA-seq reads. 
We developed an equivalence class based method that does this with significant computational 
savings over conventional methods. This was published in [Genome Biology](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-016-0970-8).

- **Post clustering differential analysis for Single Cell RNA-seq**: 
We noticed that standard pipelines of single cell RNA seq had a data snooping 
problem leading to artificially low p-values
and a selection of spurious genes as differentiating clusters of cells discovered 
in the single cell RNA-seq analysis. 
We quantified this bias and proposed solutions. This was published in [RECOMB 2019](https://link.springer.com/book/10.1007%2F978-3-030-17083-7) and 
[Cell Systems](https://www.sciencedirect.com/science/article/pii/S2405471219302698).


- **Haplotype Assembly, Codes and Spectral Algorithms**: Humans and other higher organisms are diploid, 
that is they have two copies of their genome. These two copies  are almost identical 
with some polymorphic sites and regions (less than 0.3% of the genome). 
The problem here is to estimate which of the polymorphisms are on the same copy of a chromosome. 
We attacked this in two theoretic frameworks. 
The first was noting that this was a convolutional code, and the second was posing this as a community detection problem. 
These were published at [ISIT-2015](https://arxiv.org/abs/1502.01975)
and [ICML-2016](https://arxiv.org/abs/1602.03828).


Some other problems I've dabbled with include trying to understand deep neural nets in the context of
information theoretic and computational genomics problems (like the problem of finding DNA motifs binding
with proteins), and building deep learning models to predict where CRISPR-CAS9 guides cut a DNA sequences. 