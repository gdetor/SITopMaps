#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2010-2022 Georgios Is. Detorakis (gdetor@gmail.com)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the <organization> nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pylab as plt
from matplotlib.colors import LightSource


def plot_(x, index=0, n_samples=10, labels=None, title="unknown", ylim=None):
    """ Plots the peak activity of neural fields and excitatory part of
    kernel.
    """
    fig = plt.figure(figsize=(30, 15))
    for i in range(n_samples):
        ax = fig.add_subplot(2, 5, i+1)
        ax.plot(x[i][:, index], lw=0.5, label=labels[i])
        ax.legend()
        ax.set_title(title)
        if ylim is not None:
            ax.set_ylim([60, 80])


def plot_stim(X):
    """ Plots the stimulation """
    fig = plt.figure(figsize=(30, 15))
    for i, x in enumerate(X):
        ax = fig.add_subplot(2, 5, i+1)
        y = x.mean(axis=0)
        # im = ax.imshow(x.mean(axis=0).reshape(16, 16))
        # fig.colorbar(im)
        ax.hist(y, bins=50)
        print("Experiment %d, mu: %f, std: %f" % (i, y.mean(), y.std()))


def plot_init_weights(X):
    """ Plots the initial weights """
    fig = plt.figure(figsize=(30, 15))
    norms = []
    for i, x in enumerate(X):
        x = x.reshape(32, 32, 16, 16)
        norm = np.zeros((32, 32))
        for j in range(32):
            for k in range(32):
                norm[j, k] = np.linalg.norm(x[j, k])

        norms.append(norm)
        ax = fig.add_subplot(2, 5, i+1)
        im = ax.imshow(norm)
        fig.colorbar(im)
        print(np.linalg.norm(x))
    return norms


def plot_norms_hist(X):
    """" Plots the histogram of the norms """
    fig = plt.figure(figsize=(30, 15))
    for i, x in enumerate(X):
        ax = fig.add_subplot(2, 5, i+1)
        mu = x.mean()
        ax.hist(x.flatten(), bins=50)
        ax.axvline(mu, c='k')


def plot_3d_hist(Z, Rn=16):
    """ 3D plot of stimulus """
    X, Y = np.meshgrid(np.linspace(0, 1, Rn), np.linspace(0, 1, Rn))
    ls = LightSource(270, 45)

    fig = plt.figure(figsize=(30, 15))
    for i, z in enumerate(Z):
        z = z.mean(axis=0).reshape(Rn, Rn)
        ax = fig.add_subplot(2, 5, i+1, projection='3d')
        rgb = ls.shade(z,
                       cmap=plt.cm.gist_earth,
                       vert_exag=0.1,
                       blend_mode='soft')
        ax.plot_surface(X,
                        Y,
                        z,
                        rstride=1,
                        cstride=1,
                        facecolors=rgb,
                        linewidth=0,
                        antialiased=False,
                        shade=False)


if __name__ == '__main__':
    n_samples = 8
    x = []
    for i in range(1, n_samples+1):
        fname = "./data/stats"+str(i)+".npy"
        x.append(np.load(fname))

    labels = [i+1 for i in range(n_samples)]
    plot_(x, index=0, n_samples=n_samples, labels=labels, title="Vmax")
    plot_(x, index=1, n_samples=n_samples, labels=labels, title="Lemax")
    plot_(x, index=2, n_samples=n_samples, labels=labels, title="Wnorm",
          ylim=1)
    # plot_(x, index=3, labels=labels, title="Imax")

    # S = []
    # for i in range(1, 11):
    #     S.append(np.load("data/samples"+str(i)+".npy"))
    # plot_3d_hist(S)
    # plot_stim(S)

    W = []
    for i in range(1, n_samples+1):
        W.append(np.load("./data/init_weights"+str(i)+".npy"))
    N = plot_init_weights(W)
    plot_norms_hist(N)
    plt.show()
