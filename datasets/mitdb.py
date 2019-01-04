import os
from glob import glob
import urllib3
import pandas as pd
from bs4 import BeautifulSoup as BSoup
import h5py
import wfdb as wf
import numpy as np
from scipy import signal as ss
from matplotlib import pyplot as plt

def download_db():
    
    extensions = ['atr', 'dat', 'hea']
    the_path = 'https://www.physionet.org/physiobank/database/mitdb/'
    
    # Guardo en data/
    savedir = 'data/mitdb'
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # Formato de guardado
    savename = savedir + '/{}.{}'

    # Encuentro todos los archivos importantes del sitio
    http = urllib3.PoolManager()
    response = http.request('GET', the_path)
    soup = BSoup(response.data)

    # Encuentro todos los links a archivos .dat 
    hrefs = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        # Descargo los archivos con los marcadores dados
        if href[-4::] == '.dat':
            hrefs.append(href[:-4])
        
    # Path al archivo en internet
    down_path = the_path + '{}.{}'

    for data_id in hrefs:
        for ext in extensions:
            webpath = down_path.format(data_id, ext)
            http = urllib3.PoolManager()
            datafile = http.request('GET', webpath)

            # Guardo localmente
            filepath = savename.format(data_id, ext)
            with open(filepath, 'wb') as out:
                out.write(datafile.data)
    
    print ('Se descargaron ' + str(len(hrefs)) + ' archivos de datos')

def get_records():

    # Descargo si no existe
    if not os.path.isdir('data/mitdb'):
        print ('Descargando mitdb ecg database...')
        download_db()
        print ('Descarga terminada')
        
    # Hay 3 archivos por record
    # *.atr es uno de ellos
    paths = glob('data/mitdb/*.atr')

    # Elimino la extensión
    paths = [path[:-4] for path in paths]
    paths.sort()

    return paths

def get_beats(annotation):

    types = np.array(annotation.symbol)
    
    beat_symbols = ['N', 'L', 'R', 'B', 'A',
        'a', 'J', 'S', 'V', 'r',
        'F', 'e', 'j', 'n', 'E',
        '/', 'f', 'Q', '?']
    
    ids = np.in1d(annotation.symbol, beat_symbols)

    # Me quedo con las posiciones
    beats = annotation.sample[ids]

    return beats


def get_normal_beats(annotation):
    
    types = np.array(annotation.symbol)
    where = np.where((types == 'N'))
    
    normal_beats = annotation.sample[where]
    
    return normal_beats

def convert_input(channel, beats):
    # Me quedo con todo los latidos

    # Creo una señal con deltas en los latidos
    dirac = np.zeros_like(channel)
    dirac[beats] = 1.0

    # Uso la ventana de hamming para la campana
    width = 36
    filter = ss.hamming(width)
    gauss = np.convolve(filter, dirac, mode = 'same')

    return dirac, gauss

def make_dataset(records, width, savepath):

    signals, labels = [], []

    # Recorro los archivos
    for path in records:
        print ('Processing file:' + path)
        data, field = wf.rdsamp(path)
        annotations = wf.rdann(path, 'atr')

        # Convierto cada canal en datos y labels
        signal, label = convert_data(data, annotations, width)

        # Acumulo
        signals.append(signal)
        labels.append(label)

    # Convierto todo en un np.array
    signals = np.vstack(signals)
    labels = np.vstack(labels)
    
    # En este caso descarto los deltas. ¡VER QUE PASA SI ENTRENAMOS CON LOS DELTA!
    labels = labels[:, 1, :]

    # Guardo en forma de diccionario
    np.save(savepath, {'signals' : signals,
                       'labels'  : labels })

def convert_data(data, annotations, width):
    
    signals, labels = [], []
    
    beats = get_beats(annotations)

    # Convierto ambos canales
    for it in range(2):
        channel = data[:, it]
        dirac, gauss = convert_input(channel, beats)
        # Junto los labesl
        label = np.vstack([dirac, gauss])

        # Ventana movil
        sta = 0
        end = width
        stride = width
        while end <= len(channel):
            # Me quedo con una ventana
            s_frag = channel[sta : end]
            l_frag = label[:, sta : end]

            # Acumulo
            signals.append(s_frag)
            labels.append(l_frag)

            # Paso a la ventana siguiente
            sta += stride
            end += stride

    # Convierto a np.array
    signals = np.array(signals)
    labels = np.array(labels)

    return signals, labels

def create_datasets():

    # Preparo los archivos
    records = get_records()

    # Mezclo los archivos
    np.random.seed(666)
    np.random.shuffle(records)

    # Tamaño de la ventana
    width = 200

    # Armo el set de entrenamiento
    make_dataset(records[:30], width, 'data/training')

    # Armo el set de validacion
    make_dataset(records[30 : 39], width, 'data/validation')

    # Armo el set de testeo
    make_dataset(records[39 : 48], width, 'data/test')

