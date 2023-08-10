# speech_detection_python

The primary goal of this project was to devise a method for detecting instances when a human is speaking in an audio stream. Traditional methods, such as decibel thresholding, proved inadequate as they might misinterpret non-speech sounds, like blowing into a microphone, as human speech.
During my preliminary research, I observed that the frequency spectrum of the human voice showed notable similarities across different individuals. However, there were also discernible shifts in the spectrum based on factors such as gender, with female voices generally appearing higher in frequency than male voices.
Despite these variations, I was able to devise a method to measure the similarity between two spectra. This involves convolving the two signals and identifying the point of maximum correlation.
By accumulating data from numerous human voices, I crafted a reference spectrum. This now enables me to compare any audio signal against this reference, effectively determining the presence of human speech.

## why not ML
In the landscape of audio processing, especially in the domain of speech detection, machine learning models have gained significant traction due to their ability to adapt and learn from vast amounts of data. 
However, when benchmarked against this FFT-based method, some key distinctions arise.

### Efficiency and Speed: The proposed method, which employs two iterations of FFT (one for spectrum calculation, second for fast convolution algorithm) on a small chunk of audio data, is computationally less intensive than training and interference of complex machine learning models. This makes it faster and more real-time friendly, especially on devices with limited computational resources.
### Simplicity: While machine learning models require extensive data preprocessing, feature extraction, and model tuning, this method is straightforward. With just a couple of FFT operations, the desired result is achieved without the intricacies associated with deep learning frameworks or extensive training datasets.
### Dependability: The FFT-based approach is deterministic. When provided with similar inputs, it will always produce the same outputs, unlike machine learning models which can sometimes yield unpredictable results due to their inherent stochastic nature.
![Correlation plots](https://github.com/JRazek/speech_detection_python/blob/master/human_vs_non_human_spectrums.png?raw=true)
