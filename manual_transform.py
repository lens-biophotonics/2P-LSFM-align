import os

import PySimpleGUI as sg
import tifffile as tiff
import numpy as np
from scipy.interpolate import interpn
import scipy.ndimage

sg.ChangeLookAndFeel('Dark')
sg.SetOptions(element_padding=(0, 0))

layout = [
    [sg.Text('T-z')],
    [sg.InputText('0')],
    [sg.Text('T-y')],
    [sg.InputText('0')],
    [sg.Text('T-x')],
    [sg.InputText('0')],
    [sg.Text('theta-z')],
    [sg.InputText('0')],
    [sg.Text('theta-y')],
    [sg.InputText('0')],
    [sg.Text('theta-x')],
    [sg.InputText('0')],
    [sg.Text('zoom-z')],
    [sg.InputText('1')],
    [sg.Text('zoom-y')],
    [sg.InputText('1')],
    [sg.Text('zoom-x')],
    [sg.InputText('1')],
    [sg.Text('shear-yz')],
    [sg.InputText('0')],
    [sg.Text('shear-xz')],
    [sg.InputText('0')],
    [sg.Text('shear-xy')],
    [sg.InputText('0')],
    [sg.Text('input file', size=(15, 1), auto_size_text=False, justification='right'),
     sg.InputText('input file'), sg.FileBrowse()],
    [sg.Button('Start', button_color=('white', 'black'), key='Start'),
     sg.Button('Stop', button_color=('white', 'black'), key='Stop'),
     sg.Button('Transform', button_color=('white', 'springgreen4'), key='Transform')]
]

window = sg.Window("Transform", layout, default_element_size=(12, 1), text_justification='r', auto_size_text=False,
                   auto_size_buttons=False, default_button_element_size=(12, 1))
window.Finalize()
window.Element('Transform').Update(disabled=True)
while True:
    event, values = window.Read()
    print(event)
    print(values)
    if event is None:
        exit(69)
    if event == 'Start':
        raw = tiff.imread(values[12])
        x = np.linspace(1, raw.shape[2], raw.shape[2])
        y = np.linspace(1, raw.shape[1], raw.shape[1])
        z = np.linspace(1, raw.shape[0], raw.shape[0])
        new_z = np.linspace(1, raw.shape[0], 2.5 * raw.shape[0])
        ext_z = int(50 - 1.25 * raw.shape[0])

        grid = np.array(np.meshgrid(new_z, y, x, indexing='ij'))
        grid = np.moveaxis(grid, 0, -1)

        a1 = interpn((z, y, x), raw, grid, 'linear')

        a1 = np.pad(a1, ((ext_z, ext_z), (0, 0), (0, 0)), mode='constant')
        print("tiff loaded")
        window.Element('Start').Update(disabled=True)
        window.Element('Transform').Update(disabled=False)
    elif event == 'Transform':
        T = np.zeros(4)
        T[-1] = 1
        zoom = np.ones(4)
        T[:3] = np.array([np.float(values[0]), np.float(values[1]), np.float(values[2])])
        angles = np.array([np.float(values[3]), np.float(values[4]), np.float(values[5])])
        zoom[:3] = np.array([np.float(values[6]), np.float(values[7]), np.float(values[8])])
        shear = np.array([np.float(values[9]), np.float(values[10]), np.float(values[11])])
        centerT = np.r_[np.array(a1.shape) / 2, 1]
        c = np.cos(angles)
        s = np.sin(angles)

        R0 = np.array([[c[0], s[0], 0], [-s[0], c[0], 0], [0, 0, 1]])
        R1 = np.array([[c[1], 0, -s[1]], [0, 1, 0], [s[1], 0, c[1]]])
        R2 = np.array([[1, 0, 0], [0, c[2], s[2]], [0, -s[2], c[2]]])

        R = np.linalg.multi_dot([R0, R1, R2])

        tempR = np.zeros((4, 4))
        tempR[:3, :3] = R
        tempR[-1, -1] = 1
        tempT = np.eye(4, 4)
        tempT[:, -1] = centerT
        tempT2 = np.copy(tempT)
        tempT2[:3, -1] *= -1

        R = np.linalg.multi_dot([tempT, tempR, tempT2])  # rotation relative to center

        Z = zoom * np.eye(4)
        Z[-1, -1] = 1
        Z = np.linalg.multi_dot([tempT, Z, tempT2])  # zoom relative to center

        S = np.eye(4)
        S[0, 1] = shear[0]  # shear_yz
        S[0, 2] = shear[1]  # shear_xz
        S[1, 2] = shear[2]  # shear_xy
        S = np.linalg.multi_dot([tempT, S, tempT2])  # shear relative to center

        M = np.linalg.multi_dot([R, Z, S])
        M[:3, -1] += T[:3]
        affine_matrix = M

        trans_out = scipy.ndimage.affine_transform(a1, affine_matrix)
        nome = os.path.basename(values[12]).split(".")[-2]
        np.savetxt(nome + "_matrice_affine.txt", affine_matrix)
        # TZCYXS order.
        tiff.imwrite(nome + "_transformed.tiff", trans_out[None, :, None, :, :].astype(np.float32), imagej=True)
        print("saved")
    elif event == 'Stop':
        break
