import torch
from voice_dataset import VoiceData
from numpy import random
import sounddevice as sd


class VoiceTest:
    def __init__(self, model_name, conv=True):

        self.model = torch.load(model_name)
        self.model.eval()
        self.voice_data = VoiceData()
        self.conv = conv



    def test(self, idx):
        x, y = self.voice_data[idx]
        if self.conv:
            x = x.unsqueeze(0)
        x = x.unsqueeze(0)
        y_ = self.model(x)

        print("Truth: ",float(y),"\t Prediction: ",float(torch.argmax(y_)))
        self.voice_data.play(idx)

    def eval(self):
        correct = 0.0
        incorrect = []
        for i in range(len(self.voice_data)):
            print("working on number ", i, end='\r')
            x,y = self.voice_data[i]
            x = x.unsqueeze(0)
            x = x.unsqueeze(0)
            y_ = self.model(x)
            if float(y) == float(torch.argmax(y_)):
                correct += 1.0
            else:
                incorrect.append((i,float(y),float(torch.argmax(y_))))

        print("Accuracy: %",(correct/len(self.voice_data)) )
        print("Incorrect guesses:")
        for i, t, g in incorrect:
            print("index: {} \t truth {} \t guess {}".format(i,t,g))

    
    def rand(self, count):
        for _ in range(count):
            self.test(random.randint(0,len(self.voice_data)))

    def guess(self):
        s = sd.rec(12000, samplerate=8000, channels=1, blocking=True)
        st = torch.tensor(s)
        st = st.T
        st = st.unsqueeze(0)
        out = self.model(st)
        guess = float(torch.argmax(out))
        print("Guess: ", guess)
        sd.play(s, samplerate=8000, blocking=True)
        return st


if __name__ == '__main__':
    #vl = VoiceTest(model_name='linear_b10_lr5e-3_e100.mod', conv=False)
    vc = VoiceTest(model_name='conv_cpu1.mod', conv=True)
    #vl.eval()
    #vc.eval()
    vc.rand(8)