# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 17:20:19 2018

@author: Francisco
"""

import wfdb as wf
import numpy as np
from datasets import mitdb as mitdb
from matplotlib import pyplot as plt
    

def show_annotations(path):

    signal, fields = wf.rdsamp(path)
    annotation = wf.rdann(path, 'atr')

    # Get data and annotations for the first 2000 samples
    howmany = 2000
    channel = signal[:howmany, 0]

    # Extract all of the annotation related infromation
    where = annotation.sample < howmany
    samp = annotation.sample[where]

    # Convert to numpy.array to get fancy indexing access
    types = np.array(annotation.symbol)
    types = types[where]
#    types = annotation.symbol[where]

    times = np.arange(howmany, dtype = 'float') / fields.get('fs')
    plt.plot(times, channel)

    # Prepare qrs information for the plot
    qrs_times = times[samp]

    # Scale to show markers at the top 
    qrs_values = np.ones_like(qrs_times)
    qrs_values *= channel.max() * 1.4

    plt.plot(qrs_times, qrs_values, 'ro')

    # Also show annotation code
    # And their words
    for it, sam in enumerate(samp):
        # Get the annotation position
        xa = times[sam]
        ya = channel.max() * 1.1

        # Use just the first letter 
        a_txt = types[it]
        plt.annotate(a_txt, xy = (xa, ya))

    plt.xlim([0, 4])
    plt.xlabel('Time [s]')
    plt.show()

def testbench():

    # Cargo los path de los datos disponibles
    records = mitdb.get_records()
    print('Hay ' + str(len(records)) + ' records')
    
    # Elijo uno de estos
    path = records[0]
    print('Cargo el archivo: ' + path)
    
    signal, fields = wf.rdsamp(path)
    annotation = wf.rdann(path, 'atr')
    
    normal_beats = mitdb.get_beats(annotation)
    
#    show_annotations(path)

    # Voy a plotear solo los primeros 3000
    howmany = 3000
    channel = signal[:howmany, 0]
    where = normal_beats < howmany
    samp = normal_beats[where]
    
    dirac, gauss = mitdb.convert_input(channel, samp)
    
    times = np.arange(howmany, dtype = 'float') / fields.get('fs')
    
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(times, channel)

    # Grafico un X por en cada latido
    qrs_times = times[samp]
    plt.plot(qrs_times, channel[samp], "x")
    
    plt.subplot(3,1,2)
    plt.plot(times, dirac)
    
    plt.subplot(3,1,3)
    plt.plot(times, gauss)
    
testbench()