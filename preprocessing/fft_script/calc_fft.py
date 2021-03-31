import soundfile as sf
from scipy.signal import stft 
import numpy as np

window_length = 1024 # Orginal-Skript: 1024
window_overlap = 523 # Original-Skript 256
band_size = 6 #Original-Skript 32

dataset_loc = './../../data/final_pre_dataset'
out_folder = './../../data/dataset_fft_l'+str(window_length)+"_o"+str(window_overlap)+"_b"+str(band_size)

sample_rate = 16000

# Loads all files in the passed files array and calculates the sftft on them.
# Returns two dictionaries:
# feats: {filename: numpy array fft calculated on windows}
# times: {filename: array containing the times of each window} 
def load_and_calc_features (files, length=1024, overlap=256, band_size=32, sample_rate=16000, verbose=True):
    max_len = len(files)
    max_freq = length//2
    feats = {}
    times = {}
    for file in files:
        f = file.split('/')[-1]
        if verbose and len(feats) % 1000 == 0:
            print(len(feats), '/', max_len)
        data, _ = sf.read(file)
        _, samp_times, data_spec = stft(data, fs = sample_rate, window = 'blackman', nperseg=length, noverlap=overlap)
        data_spec = np.log(np.abs(data_spec) + 0.00000000001)
        data_final = np.zeros((max_freq//band_size, data_spec.shape[1] - 1))

        for i in range(0, max_freq//band_size):
            data_final[int(i),:] = np.sqrt(np.sum(np.square(data_spec[i:i+band_size,:-1]), axis=0))
        feats[f] = data_final
        times[f] = samp_times
    return feats, times


def plot_fft(fft, times, name):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    fig.patch.set_facecolor('white')
    ax.imshow(fft, origin="lower", aspect="auto")
    # use Time instead of window number. Uncomment line to use window number. 
    #ax.xaxis.set_major_formatter(lambda val, _: times[int(val)] if int(val) >= 0 and int(val) <= len(times) else '')
    plt.xlabel("Zeit")
    plt.ylabel("Frequenzband")
    plt.title("Frequenzen in {}".format(name))
    plt.tight_layout()

if __name__ == '__main__':    
    import glob
    import os
    import shutil
    import matplotlib.pyplot as plt

    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
    os.makedirs(out_folder)
    os.makedirs(out_folder + '/dev')
    os.makedirs(out_folder + '/eval')
    shutil.copyfile(dataset_loc+"/dev-labels.csv", out_folder+"/dev-labels.csv")


    print('===', 'Processing training files', '============')

    # train files
    train_files = sorted([x.replace('\\', '/') for x in glob.glob(f'{dataset_loc}/dev/*.wav')]) #.split('\\')[-1]
    max_len = len(train_files)

    # load and save train files (we could pass the full array to the function, but not everyone might have the mem space to do so)
    for i, file_name in enumerate(train_files):
        if i % 1000 == 0:
            print(i, '/', max_len)
        # load
        train_feats, train_times = load_and_calc_features([file_name], length=window_length, overlap=window_overlap, sample_rate=sample_rate, verbose=False, band_size = band_size)
        name = list(train_feats.keys())[0]
        # merge time and fft
        tmp = np.concatenate([np.expand_dims(train_times[name][:-1], axis=0), train_feats[name]], axis=0).T
        # save to csv
        np.savetxt(out_folder + '/dev/' + name.replace('.wav', '.csv'), tmp, delimiter=',')


    # plot example fft:
    #print('===', 'Plotting example fft for', name, '============')
    
    #plot_fft(train_feats[name], train_times[name], name)
    #plt.savefig(name.replace('.wav', '.png'))



    print('===', 'Processing evaluation files', '============')

    # load eval files
    eval_files = sorted([x.replace('\\', '/') for x in glob.glob(f'{dataset_loc}/eval/*.wav')]) #.split('\\')[-1]
    max_len = len(eval_files)

    # save eval files
    for i, file_name in enumerate(eval_files):
        if i % 1000 == 0:
            print(i, '/', max_len)
        eval_feats, eval_times = load_and_calc_features([file_name], length=window_length, overlap=window_overlap, band_size = band_size, sample_rate=sample_rate, verbose=False)
        name = list(eval_feats.keys())[0]
        
        tmp = np.concatenate([np.expand_dims(eval_times[name][:-1], axis=0), eval_feats[name]], axis=0).T
        
        np.savetxt(out_folder + '/eval/' + name.replace('.wav', '.csv'), tmp, delimiter=',')
        