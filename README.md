# SITopMaps

This repository contains the souce code for the article [1]. SITopMap
(Somatosensory cortex Topographic Maps) provides the fundamental tools to 
reproduce the figures from [1].

The repository is organized as:
  - **DNF-2D-SOM-REF.py**  Is the main script that implements the model and the 
     learning rule.
  - **DNF-2D-REF-Response.py** This script computes the receptive fields of the 
    model for given stimuli and feed-forward weights sets. The user should first
    run the script *DNF-2D-SOM-REF.py* to obtain the grid coordinates and the
    feed-forward weights.
  - **DNF-RF-Size.py** It computes and plots all the receptive fields of the 
    model. Again this script requires the feed-forward weights and the grid.

A precomputed set of feed-forward weights (*weights.npy*), the receptors 
coordinates (*gridxcoord.npy* and *gridycoord.npy*), and a model response of a 
resolution of 64 pixels (*model_response_64*) are included in this repository
in the directory *data/*.


### Dependencies
  - Numpy
  - Matplotlib


### Platforms where SITopMaps have been tested
  - Ubuntu 20.04.5 LTS
    - GCC 9.4.0
    - Python 3.8.10
    - x86_64


### References
  1. "A Neural Field Model of the Somatosensory Cortex: Formation, Maintenance 
    and Reorganization of Ordered Topographic Maps, Georgios Is. Detorakis and 
    Nicolas P. Rougier, PLoS ONE DOI: 10.1371/journal.pone.0040257"
