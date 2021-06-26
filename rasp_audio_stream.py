import pyaudio
import numpy as np


class AudioInputStream:
    def __init__(self, 
                 format=pyaudio.paFloat32, 
                 input_device_keyword="VoiceMeeter Output",
                 CHUNK=1024,
                 maxInputChannels=2
                 ):
        self.maxInputChannels = maxInputChannels
        self.CHUNK = CHUNK
        self.format = format
        if format is pyaudio.paFloat32:
            self.dtype = np.float32
        elif format is pyaudio.paInt16:
            self.dtype = np.int16
        # Open the stream    
        self.p = pyaudio.PyAudio()
        self.__open_stream(input_device_keyword)
    
    def get_params(self):
        params_dict = {
            "RATE": self.RATE,
            "CHUNK": self.CHUNK,
            "CHANNELS": self.CHANNELS,
        }
        return params_dict
    
    def __open_stream(self, input_device_keyword):
        self.input_device_index = None
        self.input_device_name = None
        self.devices = []
        print(f"=========================================================")
        print(f"dev. index\tmaxInputCh.\tmaxOutputCh.\tdev. name")
        
        for k in range(self.p.get_device_count()):
            dev = self.p.get_device_info_by_index(k)
            self.devices.append(dev)
            device_name = dev["name"]
            device_index = dev["index"]            
            maxInputChannels = int(dev["maxInputChannels"])
            maxOutputChannels = int(dev["maxOutputChannels"])
                
            if type(device_name) is bytes:
                device_name = device_name.decode("cp932")  # for windows
                
            print(f"{device_index}\t{maxInputChannels}\t{maxOutputChannels}\t{device_name}")
       
            if  input_device_keyword in device_name \
                    and maxInputChannels == self.maxInputChannels:
                self.input_device_index = dev["index"]
                self.input_device_name = device_name
                self.RATE = int(dev["defaultSampleRate"])
                self.CHANNELS = dev["maxInputChannels"]

        if self.input_device_index is not None:
            print(f"=========================================================")
            print(f"Input device:  {self.input_device_name} is OK.")
            print(f"\tRATE:      {self.RATE}")
            print(f"\tCHANNELS:  {self.CHANNELS}")
            print(f"\tCHUNK:     {self.CHUNK}")
            print(f"=========================================================")
        else:
            print(f"\nWarning: Input device is not exist\n")

        self.stream = self.p.open(
                format=self.format,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                output=False,
                frames_per_buffer=self.CHUNK,
                input_device_index=self.input_device_index,
            )
            
        return self

    def run(self, callback_sigproc):
        while self.stream.is_active():
            input_buff = self.stream.read(self.CHUNK)
            data = np.fromstring(input_buff, dtype=self.dtype)
            # data: []L, R, L, R, ..., L, R] => data[n_fft, 2] (data[n_fft, 0] is Left channel)
            sig = np.reshape(data, (self.CHUNK, self.CHANNELS)).T
            callback_sigproc(sig)
        self.__terminate()
        
    def __terminate(self):
        stream.stop_stream()
        stream.close()
        p.terminate()
    
def test_callback_sigproc(sig):
    print(sig.shape)
    
if __name__ == "__main__":
    ais = AudioInputStream()
    print(ais.get_params())
    ais.run(test_callback_sigproc)

