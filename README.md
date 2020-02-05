# "Fine tuning" google's yamnet network to detect train sounds 

Accompanies the blog post here: 
http://labs.laan.com/blog/building-a-train-horn-detection-neural-network.html

This was a weekend project to track / quantify the train in my neighborhood that honks a lot. 
I eventually plan to correlate with my sleep patterns, but that's another day. 

As this was a short weekend project, no care has been take around clean code / bugs / proper validation dataset, etc.

### Some background
Google released yamnet: see https://github.com/tensorflow/models/tree/master/research/audioset/yamnet 
"YAMNet is a pretrained deep net that predicts 521 audio event classes based on the AudioSet-YouTube corpus, and employing the Mobilenet_v1 depthwise-separable convolution architecture."

The yamnet model does contain a 'train' and 'train horn' class, but found it was not accurate enough on my data. If your data is one of the 521 classes, you might be able to directly use the yamnet output and not train anything. 


This notebook takes the output feature vectors from yamnet as 1024 length vectors per patch of audio. We then feed those vectors into a small network that has been trained on the data I care about ( "train horn" vs "not train horn" ) 

I created a very small dataset of about 10 wavs per class -- where the "not train horn" class is just background noise, or noises that might be confused with a train horn ( dog bark, etc ) Each training sound is augmented a few times to create more data. 

Note: I did remove a few files from the data included here for file size reasons, but only 1-2 wav files. The accuracy here will be slightly lower than what I got. You can experiment with how much data is required for your task. 
