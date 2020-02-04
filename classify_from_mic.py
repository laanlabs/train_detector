#!/usr/bin/env python
# coding: utf-8

import sys

yamnet_base = './models/research/audioset/yamnet/'
sys.path.append(yamnet_base)

import os

assert os.path.exists(yamnet_base)

import time
import numpy as np

# audio stuff 
import librosa
import soundfile as sf
import resampy
import pyaudio

# yamnet imports 
import params
import modified_yamnet as yamnet_model
import features as features_lib

# TF / keras 
#from tensorflow.keras import Model, layers
import tensorflow as tf
from tensorflow.keras.models import load_model


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


DESIRED_SR = 16000


# TODO: include my slightly modified yamnet code in this file 
# i added the 'dense_net' return
def load_yamnet_model(model_path='yamnet.h5'):
    # Set up the YAMNet model.
    params.PATCH_HOP_SECONDS = 0.1  # 10 Hz scores frame rate.
    yamnet, dense_net = yamnet_model.yamnet_frames_model(params)
    yamnet.load_weights(model_path)
    return yamnet


def load_top_model(model_path="top_model.h5"):
    return load_model(model_path)


def read_wav(fname, output_sr, use_rosa=False):
    
    if use_rosa:
        waveform, sr = librosa.load(fname, sr=output_sr)
    else:
        wav_data, sr = sf.read(fname, dtype=np.int16)
        
        if wav_data.ndim > 1: 
            # (ns, 2)
            wav_data = wav_data.mean(1)
        if sr != output_sr:
            wav_data = resampy.resample(wav_data, sr, output_sr)
        waveform = wav_data / 32768.0
    
    return waveform.astype(np.float64)


def remove_silence(waveform, top_db=15, min_chunk_size=2000, merge_chunks=True):
    """
    Loads sample into chunks of non-silence 
    """
    splits = librosa.effects.split(waveform, top_db=top_db)
    
    waves = []
    for start, end in splits:
        if (end-start) < min_chunk_size:
            continue
        waves.append(waveform[start:end])
    
    if merge_chunks and len(waves) > 0:
        waves = np.concatenate(waves)
    
    return waves


def run_models(waveform, 
               yamnet_model, 
               top_model, 
               strip_silence=True, 
               min_samples=11000):
    
    if strip_silence:
        waveform = remove_silence(waveform, top_db=10)
    
    if waveform is None:
        print('none wav?')
        return None
    
    if len(waveform) < min_samples:
        #print(" too small after silence: " , len(waveform))
        return None
    
    # predictions, spectrogram, net, patches
    _scores, _spectro, dense_out, _patches = \
        yamnet_model.predict(np.reshape(waveform, [1, -1]), steps=1)
    

    # dense = (N, 1024)
    all_scores = []
    for patch in dense_out:
        scores = top_model.predict( np.expand_dims(patch, 0) ).squeeze()
        all_scores.append(scores)
    
    if not all_scores:
        # no patches returned
        return None
    
    all_scores = np.mean(all_scores, axis=0)
    return all_scores



def run_detection_loop(input_device_index = 0):

    yamnet = load_yamnet_model()
    top_model = load_top_model()

    CHUNK = 4096 * 2

    FORMAT = pyaudio.paInt16

    DTYPE = np.int16 if FORMAT == pyaudio.paInt16 else np.float32

    CHANNELS = 1
    RATE = DESIRED_SR

    min_frames_to_process = int(DESIRED_SR * 2.5)

    p = pyaudio.PyAudio()
    p.get_device_count()

    for i in range(p.get_device_count()):
        print(p.get_device_info_by_index(i))
        print("_______")

    p.terminate()


    def dump_wav(arr, fname):    
        librosa.output.write_wav(fname, arr, DESIRED_SR)


    if not os.path.exists("logs"):
        os.makedirs("logs")

    if not os.path.exists("detections"):
        os.makedirs("detections")        

    if not os.path.exists("train_wavs"):
        os.makedirs("train_wavs")
        os.makedirs("train_wavs/high")
        os.makedirs("train_wavs/mid")
        os.makedirs("train_wavs/low")

    def log_line(line, type='info'):
        timestr = time.strftime('%a, %d %b %Y %H:%M:%S', time.localtime() )
        log_file.write("{:<30} [{:<8}] {}\n".format(timestr, type, line))
        log_file.flush()

    def log_detection(score, wav_file=""):
        timestr = time.strftime('%a %d %b %Y %H:%M:%S', time.localtime() )
        timestr2 = str(time.time())
        score = np.round(score, 3)
        csv_file.write("{},{},{},{}\n".format(timestr, timestr2, score, wav_file))
        csv_file.flush()



    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=input_device_index,
                    frames_per_buffer=CHUNK)

    frames = []
    chunks_required = int(np.ceil(min_frames_to_process // CHUNK))

    MIN_NOISE = 0.1
    NOISE_MEAN_SCALE = 30.0

    top_db = 18

    MIN_SAMPLES_TO_RUN_NN = 5500

    train_scores = []
    train_times = []

    all_sounds = []
    all_sounds_times = []

    verbose = 0

    timestr = time.strftime('%a_%d_%b_%Y_%H-%M-%S', time.localtime())

    log_name = "logs/{}.txt".format(timestr)
    log_file = open(log_name, 'w')

    csv_name = "detections/{}.csv".format(timestr)
    csv_file = open(csv_name, 'w', encoding='utf-8')

    last_web_update = time.time()
    last_ping_time = time.time()


    try:
        
        while True:
            
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
            except OSError:
                print(" __ overflow")

            arr = np.frombuffer(data, dtype=DTYPE)
            arr = arr.astype(np.float32)
            arr = arr / 32768.0
            arr = arr * 1.25 
            
            frames.append(arr)
            
            if len(frames) > chunks_required:
                frames.pop(0)
            
            if len(frames) >= chunks_required:
                
                wave_arr = np.concatenate(frames)
                noise_mean = np.abs(wave_arr).mean() * NOISE_MEAN_SCALE
                
                if noise_mean < MIN_NOISE:
                    continue
                
                wave_arr = remove_silence(wave_arr, top_db=top_db)
                
                if wave_arr is None:
                    continue 
                    
                # hack .. double the wave if too short
                if len(wave_arr) > MIN_SAMPLES_TO_RUN_NN//2 and len(wave_arr) < MIN_SAMPLES_TO_RUN_NN:
                    wave_arr = np.concatenate((wave_arr, wave_arr))
                
                scores = None
                
                noise_mean = np.abs(wave_arr).mean() * NOISE_MEAN_SCALE
                
                if noise_mean < MIN_NOISE:
                    continue
                
                if wave_arr is not None and len(wave_arr) >= MIN_SAMPLES_TO_RUN_NN:
                    # not sure what the min size is for yamnet -- somewhere around 5k ? 
                    
                    scores = run_models(wave_arr, yamnet, top_model, strip_silence=False)

                    scores_text = "None"

                    if scores is not None:
                        # little bar for train score
                        scores_text = "="*int(scores[1]*50)
                    
                    #log_line("detection:  {:<10}  {}".format(len(wave_arr), scores_text))
                    
                    if scores is not None:
                        
                        train_score = scores[1]
        
                        if train_score > 0.05:
                            print("  detection: ", train_score)
                            rounded_score = int(train_score * 100)
                            train_scores.append(train_score)
                            train_times.append(time.time())
                            
                            if train_score > 0.7:
                                folder = "high"
                            elif train_score > 0.35:
                                folder = "mid"
                            else:
                                folder = "low"
                            
                            wav_out_path = "train_wavs/{}/{}_{}.wav".format(folder, len(train_scores), rounded_score)
                                
                            dump_wav(wave_arr, wav_out_path)


                
    except KeyboardInterrupt as e:
        print(" ____ interrupt ___")
        stream.stop_stream()
        stream.close()
        p.terminate()
        log_file.close()
        
    except Exception as e:
        stream.stop_stream()
        stream.close()
        p.terminate()
        log_file.close()
        print(" err" , str(e))
        raise e


if __name__ == '__main__':
    input_device_index = 0
    if len(sys.argv) > 1:
        input_device_index = int(sys.argv[1])
    print(" --- Using input device: ", input_device_index)
    run_detection_loop(input_device_index=input_device_index)



