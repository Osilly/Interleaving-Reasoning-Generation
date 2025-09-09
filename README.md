# Interleaving Reasoning Generation
The official repo for "Interleaving Reasoning for Better Text-to-Image Generation".



<p align="center">
       &nbsp&nbsp📑 <a href="https://arxiv.org/abs/2509.06945">Paper</a>&nbsp&nbsp
</p>

The datasets, code and weights will be released, stay tuned!

## Timeline

- [2025/03/09] Our IRG paper ([Interleaving Reasoning for Better Text-to-Image Generation](https://arxiv.org/abs/2509.06945)) can be accessed in arXiv!

## Overview

![](figs/overview.png)

> As shown in (a), we illustrate an example of Interleaving Reasoning Generation (IRG).  Given a prompt, the model first produces a text‑based reasoning process and then generates an image conditioned on that reasoning. Next, building upon the initial image, the model reflects on how to improve its quality and produces a refined image through this reflection process. IRG can substantially enhance image generation quality.  For instance, in the top case of (a), IRG improves upon the previous generated image via multi‑turn reasoning, enhancing rendering textures, shadow realism, and other visual properties.  In the bottom case of (a), IRG significantly improves fine‑grained details, such as the delicate structures of fingers—highlighted within the red box (as detailed in (b)). As shown in (c), compared to current SoTA models, our proposed IRG achieves clearly superior performance across multiple mainstream T2I benchmarks.

## IRG Case

![](figs/big_case.png)

## Case Comparison 

![](figs/compare.png)

## Pipeline

![](figs/pipeline.png)

> Overview of our proposed IRG training and inference pipeline. IRG learns the text-based thinking process and the complete high-quality image generation pipeline under six decomposed learning modes. During inference, we introduce a dedicated CFG condition design for IRG’s improved image generation steps. 
