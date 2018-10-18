# darkdisk
Project: <a href="https://arxiv.org/abs/1808.05603">Using Gaia DR2 to Constrain Local Dark Matter Density and Thin Dark Disk</a>

Author: <a href="http://inspirehep.net/author/profile/Shing.Chau.Leung.1">John Leung</a>

Date: 2017-2018

## Feature:

Select gaia astrometric data (stellar kinematics). Perform cuts to select stars of interest as tracers. Extract information concerning Milky Way gravitational potential by solving Poison-Jeans equation using the tracer stars. This program partly contributes to the paper "*Using Gaia DR2 to Constrain Local Dark Matter Density and Thin Dark Disk*", <a href="https://arxiv.org/abs/1808.05603">1808.05603</a> by Jatan Buch, John Shing Chau Leung, JiJi Fan.

The python notebooks serve as a narration of high the data is generated from the raw Gaia data. Readers can follow the reading sequence of *gaia_cut* → *exclusion_plot* → *emcee_gaia*. The computationally simplifed version of the main result is featured in emcee_gaia.ipynb.

Result for the likelihood-frequentist exclusion contour.
![95% exclusion contour in the dark disk parameter plane.](Plots/exclusion_contour.png?raw=true "Title" width="200" height="400")

Result for the MCMC posterior probabilities.
![Posteriors for Dark matter and baryons with MCMC.](Plots/emcee.png?raw=true "Title" width="400" height="400")

## Acknowledgement:

This script uses and modifies <a href="https://github.com/jobovy">Jo Bovy</a>'s gaia-tools. For installation instruction for gaia_tools, please refer to: <a href="https://github.com/jobovy/gaia_tools">https://github.com/jobovy/gaia_tools</a>.
