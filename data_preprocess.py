import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import glob
from DRT_utilis import Simple_run
from tqdm import tqdm
from scipy import interpolate
import itertools
import pickle

def plot_test_all(label):
    # plot all test data, including EIS and LSV, to check the data quality
    files = glob.glob(f'../txt_data/{label}/*.txt')
    for file in files:
        plt.figure()
        fig_name = file.split('/')[-1][:-4]
        cell = pd.read_csv(file, delim_whitespace=True)
        if label == 'EIS':
            Z_Re = cell['Re(Z)/Ohm']
            Z_Im = -cell['-Im(Z)/Ohm']
            plt.plot(Z_Re, -Z_Im, label=fig_name)
            plt.plot(Z_Re, -Z_Im, "o", color="black")
            plt.legend()
        elif label == 'LSV':
            plt.plot(cell['<I>/mA'], -cell['Ewe/V'])
        plt.savefig(f'figs/{label}/'+fig_name+'.jpg')
        plt.close()

def plot_EIS(Z_exp, Z_model, fig_name, model_label='DRT'):
    # function to plot the DRT impedance and experimentally collected data points
    plt.figure()
    plt.plot(Z_exp.real, -Z_exp.imag, "o", markersize=10, color="black", label="exp")
    plt.plot(Z_model.real, -Z_model.imag, linewidth=4, color="blue", label=model_label)
    plt.legend(frameon=False, fontsize = 15)
    xmin = int(min(Z_model.real))
    plt.xlim([xmin, xmin+7])
    plt.ylim([-3.5, 3.5])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel(r'$Z_{\rm re}/\Omega$', fontsize = 20)
    plt.ylabel(r'$-Z_{\rm im}/\Omega$', fontsize = 20)
    plt.savefig(f'figs/EIS/{fig_name}.tif', dpi=300, bbox_inches='tight')
    plt.close()

def txt2zview(Z_data, z_file_name):
    # convert the txt file into zview data file, thus using the ECM model to fit the impedance
    with open(z_file_name, "w") as z_file:
        z_file.write('"ZPlotW Data File: Version 3.1"\n')
        z_file.write('"Modified Data"\n"Sweep Frequency: Control Voltage"\n"Date: 12-6-2023     Time: 20:46:8\n""\n""\n')
        z_file.write(f'"{z_file_name}"\n"Frequency"\n0,2,0,1,0.1,65000\n36\n')
        column_name = "  Freq(Hz)    Ampl     Bias   Time(Sec)   Z'(a)    Z''(b)    GD   Err   Range"
        z_file.write(f'"{column_name}"\n')
        for i in range(len(Z_data)):
            freq = Z_data['freq/Hz'].iloc[i]
            Z_re, Z_im = Z_data['Re(Z)/Ohm'].iloc[i], -Z_data['-Im(Z)/Ohm'].iloc[i]
            z_file.write('{0:.6e}, {1:.4e}, {2:.4e}, {3:.6e}, {4:.4e}, {5:.4e}, {6:.4e}, {7}, {8}\n'.format(freq, 0.0, 0.0, 3.0*(i+1), Z_re, Z_im, 0.0, 0, 0))

def convert_files():
    # call the function txt2zview() to convert all txt files into zview data file
    files = glob.glob(f'../txt_data/EIS/*.txt')
    for org_file in tqdm(files):
        z_file_name = org_file[:-4]+'.z'
        Z_data = pd.read_csv(org_file, delim_whitespace=True)
        txt2zview(Z_data, z_file_name)

def outlier_process(freq_ref):
    # using the DRT model to fit the impedance and recover it by excluding the outlier datapoints
    # outlier index for the specified EIS test
    outlier_index = {'1 (5)': [25, 26, 30, 34, 35], '2 (5)': [35], '2 (6)': [23, 24, 25]+list(range(27, 36)), '3 (2)': [34], '4 (5)': [34, 35], '4 (6)': [32, 33, 34], '4 (7)': [34], 
                    '5 (1)': [21],'6 (1)': [34, 35], '6 (5)': [28], '6 (6)': list(range(31, 34)), '6 (7)': [28, 34, 35], 
                    '7 (1)': [33, 34], '7 (2)': [25, 27, 30, 31, 32, 34, 35], '7 (3)': [35], '7 (4)': [34, 35], '7 (5)': [29]+list(range(31, 35)), '7 (7)': [34], 
                    '8 (1)': list(range(24, 29))+list(range(31, 35)), '8 (3)': [23, 32]+list(range(25, 30)), '8 (5)': [24], '8 (7)': [19]+list(range(27, 36)), 
                    '9 (1)': [31, 32, 34], '9 (2)': [28, 29, 30, 31], '9 (4)': [25], '9 (6)': [], '10 (2)': [33, 34, 35], '10 (3)': [27], '10 (5)': [30], '10 (6)': [35], 
                    '11 (2)': [34, 35], '11 (3)': list(range(28, 36)), '11 (4)': [34], '11 (6)': [34, 35], '12 (3)': [22, 29, 31, 32, 33, 35], '15 (4)': [], '17 (3)': [], 
                    '20 (1)': [23, 24, 28, 30, 33, 34], '20 (3)': [], '21 (3)': [35], '22 (5)': [29], '23 (3)': [], '23 (4)': [], '23 (5)': [],  '23 (6)': [], 
                    '24 (1)': list(range(28, 31)), '24 (3)': [33, 35], '24 (4)': [], '25 (1)': [31], '25 (2)': [34], '25 (3)': [22], '25 (4)': [22], '25 (6)': [], 
                    '26 (1)': [21, 23], '26 (2)': [], '26 (3)': [18, 20], '26 (6)': [25, 27, 28], '27 (1)': [22, 25]+list(range(27, 35)), '27 (3)': [31], '27 (4)': [35], '27 (5)': [34], 
                    '28 (1)': [30], '28 (2)': [], '28 (3)': [35], '28 (5)': [], '29 (1)': [33], '29 (2)': [32], '29 (4)': [], '31 (4)': [35], '31 (5)': [], 
                    '32 (1)': [], '32 (2)': [33], '33 (2)': [34], '33 (4)': [32], '33 (7)': [28], '34 (3)': [33], '34 (5)': [26], '35 (2)': [30], '35 (5)': [33], '35 (6)': list(range(32, 36)), 
                    '36 (2)': [19], '36 (5)': [35], '37 (4)': [27], '37 (7)': [33], '38 (7)': [31], '39 (2)': [20], '39 (4)': [33, 34, 35], '39 (7)': [26, 28, 30, 32], 
                    '40 (6)': [31], '40 (7)': [33], '41 (1)': [23, 26, 29], '41 (3)': [], '41 (4)': [32, 33, 34], '41 (5)': [30], '41 (7)': [34, 35], 
                    '42 (1)': [22, 28, 29, 35], '42 (2)': [34, 35], '42 (3)': [34], '42 (4)': [20, 35], '42 (5)': [32, 34, 35], '42 (7)': [28, 29, 31, 32]}
    # the following test sets are corrected by ECM fitting, so the DRT model will not be implemented for those
    # outlier_supp = {'3 (7)': [26, 28, 33, 35], '4 (1)': [30, 33], '5 (2)': [29, 32, 33], '5 (6)': [29, 34], '6 (3)': [27, 29], '12 (2)': list(range(32, 36)), '29 (6)': [], '42 (6)': [27], }
    folder = '../txt_data/EIS/'
    for key, ind in tqdm(outlier_index.items()):
        fname = key+'.txt'
        cell = pd.read_csv(folder+fname, delim_whitespace=True)
        mask = np.ones(len(cell), dtype=bool)
        mask[ind] = False
        freq = cell['freq/Hz'].to_numpy()
        Z_exp = cell['Re(Z)/Ohm'].to_numpy() - 1j*cell['-Im(Z)/Ohm'].to_numpy()
        EIS_data = {'freq': freq[mask], 'Z_exp':Z_exp[mask]}
        Z_model = Simple_run(EIS_data, freq_ref, rbf_type='Piecewise Linear',)
        plot_EIS(Z_exp, Z_model, key+'_DRT', model_label='DRT')
        Z_model = pd.DataFrame({'freq/Hz': freq_ref, 'Re(Z)/Ohm': Z_model.real, '-Im(Z)/Ohm': -Z_model.imag})
        folder_corrected = '../txt_data/EIS_corrected/'
        Z_model.to_csv(f'{folder_corrected}{key}_DRT.txt', sep=' ', index=False)

def ECM_fit(result_file, freq_ref):
    # the function to recover the impedance data points by ECM fiting results. The ECM either consits of two R//CPE or three R//CPE elements in connection with a Ohm resistor $R_{ohm}$ and inductance $L$
    # the ECM behaves lile L-R-[R1//CPE1]-[R2//CPE2]-[R3//CPE3]
    # the parameters in the column are listed as L, R, R1, CPE1_T, CPE1_P, R2, CPE2_T, CPE2_P, R3, CPE3_T, CPE3_P
    result = pd.read_csv(result_file, index_col=0)
    omega = 2*np.pi*freq_ref
    for name in tqdm(result.columns.values):
        paras = result[name]
        paras = paras[~np.isnan(paras)]
        if len(paras) == 8:
            # two R//CPE unit
            Z_model = 1j*omega*paras['L'] + paras['R0'] + 1/(1/paras['R1'] + paras['CPE1_T']*(1j*omega)**paras['CPE1_P']) + 1/(1/paras['R2'] + paras['CPE2_T']*(1j*omega)**paras['CPE2_P'])
        elif len(paras) == 11:
            Z_model = 1j*omega*paras['L'] + paras['R0'] + 1/(1/paras['R1'] + paras['CPE1_T']*(1j*omega)**paras['CPE1_P']) + 1/(1/paras['R2'] + paras['CPE2_T']*(1j*omega)**paras['CPE2_P']) + \
                1/(1/paras['R3'] + paras['CPE3_T']*(1j*omega)**paras['CPE3_P'])
        
        Z_data = pd.DataFrame({'freq/Hz': freq_ref, 'Re(Z)/Ohm': Z_model.real, '-Im(Z)/Ohm': -Z_model.imag})
        filename = '../txt_data/EIS_corrected/'+name+'_ECM.txt'
        Z_data.to_csv(filename, sep=' ', index=False)
        
        # read the experiemntally measured data points and visualize the ECM together
        df = pd.read_csv(f'../txt_data/EIS/{name}.txt', delim_whitespace=True)
        Z_exp = df['Re(Z)/Ohm'].to_numpy() - 1j*df['-Im(Z)/Ohm'].to_numpy()
        plot_EIS(Z_exp, Z_model, name+'_ECM', model_label='ECM')

def EIS_corrected_combine():
    # combine all EIS data into a csv or npy format for later use
    files = glob.glob('../txt_data/EIS/*.txt')
    files.sort(key=lambda s: int(os.path.basename(s)[:-8])) # glob returns list with arbitrary order
    corrected_files = glob.glob('../txt_data/EIS_corrected/*.txt')
    corrected_file_list = [os.path.basename(s).split('_')[0] for s in corrected_files]
    assert len(np.unique(corrected_file_list)) == len(corrected_file_list), "There are some duplicate files, please remove them!"
    df = pd.read_csv(files[0], delim_whitespace=True)
    freq_ref = df['freq/Hz'].to_numpy()

    EIS_data = np.zeros((len(files), len(freq_ref)*2))
    corrected_cnt = 0
    for i, file in enumerate(files):
        test_label = os.path.basename(file)[:-4]
        if test_label in corrected_file_list:
            corrected_file = glob.glob(f'../txt_data/EIS_corrected/{test_label}_*.txt')
            assert len(corrected_file) == 1, f"Duplicate files {corrected_file[0]} exist, please remove them!"
            cell = pd.read_csv(corrected_file[0], delim_whitespace=True)
            corrected_cnt += 1
        else:
            cell = pd.read_csv(file, delim_whitespace=True)
        EIS_data[i] = np.hstack((cell['Re(Z)/Ohm'].to_numpy(), -cell['-Im(Z)/Ohm'].to_numpy()))
    
    print(f'Corrected EIS data count is {corrected_cnt}!')
    print('Save the EIS_data and freq in the .npy format for later use!')
    np.save('./data/EIS_data.npy', EIS_data)
    np.save('./data/freq.npy', freq_ref)

def linear_fit(x, y, x_sample, deg=1):
    # run linear fit and return the new_y at x data points
    z = np.polyfit(x, y, deg=1)
    p = np.poly1d(z)

    return p(x), p(x_sample)

def dump_file(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)

def view_LSV(current, volt, I_sample, E_sample, test_label):
    plt.figure()
    plt.plot(current, volt, 'b-', label='exp')
    plt.plot(I_sample, E_sample, 'ro', label='sample')
    plt.legend()
    plt.xlabel(r'${\rm I}/(\rm {A/cm^2})$')
    plt.ylabel(r'${\rm E/V}$')
    plt.text(0.7, 1.0, f'test_label: {test_label}', fontsize=12)
    plt.savefig(f'figs/LSV/{test_label}_sample.png', dpi=300)
    plt.close()

def sample_LSV_curve():
    # Sample data points in the measured LSV curve by interpolation. 
    # Caution: Some measured curves show oscillations and we should pay more attention
    outlier_ind = ['1 (5)', '4 (1)', '4 (5)', '4 (7)', '5 (1)', '5 (6)', '5 (7)', '6 (3)', '6 (6)', '7 (2)', '7 (7)', 
                   '8 (6)', '8 (7)', '9 (1)', '11 (3)', '15 (4)', '24 (3)', '27 (1)', '29 (6)', '34 (1)', '40 (5)']
    files = glob.glob('../txt_data/LSV/*.txt')
    files.sort(key=lambda s: int(os.path.basename(s)[:-8])) # glob returns list with arbitrary order
    sample_volt = np.arange(1.0, 0.0, -0.1)
    sample_current = []
    E_I_array = []
    for file in files:
        file_name = file.split('/')[-1][:-4]
        cell = pd.read_csv(file, delim_whitespace=True)
        I = cell['<I>/mA'].to_numpy()/1000/0.07065 # convert into $A/{cm^2}$
        E = -cell['Ewe/V'].to_numpy()
        E_I_array.append({file_name: np.vstack((E, I))})
        # use the scipy.interpolate.interp1d instead of numpy.interpolate. 
        # To avoid the error by sampling data at 0V, pass the varaible fill_value
        f_interp = interpolate.interp1d(E[2:], I[2:], fill_value="extrapolate")
        I_sample = f_interp(sample_volt)
        if file_name in outlier_ind:
            # process the LSV curve that show oscillations
            E_split = np.array_split(E[2:], len(sample_volt))
            I_split = np.array_split(I[2:], len(sample_volt))
            I_fit, I_sample = [], []
            for I_sub, E_sub, E_data in zip(I_split, E_split, sample_volt):
                I_sub_fit, I_data_fit = linear_fit(E_sub, I_sub, E_data)
                I_fit.append(I_sub_fit)
                I_sample.append(I_data_fit)
            print(f'The error of interpolated currents between two methods for {file_name} are:\n')
            print(f_interp(sample_volt) - I_sample)
            plt.figure()
            plt.plot(I, E, label='exp')
            plt.plot(list(itertools.chain(*I_fit)), E[2:], 'r--', label='corrected')
            plt.legend()
            plt.savefig(f'figs/LSV/{file_name}_corrected.tif')
            plt.close()
        view_LSV(I, E, I_sample, sample_volt, file_name)
        sample_current.append(I_sample)
    I_matrix = np.array(sample_current)
    print(I_matrix.shape)
    # save as .npy format, making the data load easier
    np.save('./data/I_sample.npy', I_matrix)
    dump_file('./data/E_I_array.pkl', E_I_array)

if __name__=="__main__":
    # plot_test_all('LSV')
    # define the reference frequency points to recover the EIS data
    freq_ref = np.array([1.0000371E+005, 6.7389773E+004, 4.5408992E+004, 3.0600504E+004, 2.0616836E+004, 1.3899199E+004, 9.3622354E+003, 6.3077358E+003, 4.2513608E+003, 2.8659180E+003, 1.9306168E+003, 1.3012513E+003, 
            8.7696313E+002, 5.9094580E+002, 3.9808899E+002, 2.6819781E+002, 1.8075183E+002, 1.2183264E+002, 8.2074852E+001, 5.5300816E+001, 3.7278458E+001, 2.5120602E+001, 1.6922356E+001, 1.1405126E+001, 
            7.6894464E+000, 5.1772671E+000, 3.4916177E+000, 2.3521037E+000, 1.5853271E+000, 1.0674969E+000, 7.2004646E-001, 4.8488677E-001, 3.2671914E-001, 2.2013262E-001, 1.4845580E-001, 9.9951774E-002])
    # outlier_process(freq_ref)
    # convert_files()
    # ECM_fit('../txt_data/EIS/ECM fitresults.csv', freq_ref)
    # EIS_corrected_combine()
    sample_LSV_curve()

