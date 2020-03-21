import matplotlib.pyplot as plt
import json


class Grapher:

    def __init__(self, filename, name='Model'):
        self.gpus = ['1','2','3','4']
        bth_int = [10, 20, 30, 50, 70, 100, 150, 200, 300, 400, 500, 600]
        self.batch_sizes = [str(b) for b in bth_int]
        with open(filename, 'r') as f:
            s = f.read()
        self.data_dict = json.loads(s)


    def get_gpu(self, gpu):
        gpu_val = self.gpus[gpu]
        return [self.data_dict[gpu_val][batch]['epoch_time'] for batch in self.batch_sizes]

    def get_batch(self, batch):
        batch = str(batch)
        return [self.data_dict[gpu][batch]['epoch_time'] for gpu in self.gpus]

    def plot_gpus(self, gpus):
        
        # styling
        plt.figure(facecolor='white', edgecolor='black', figsize=(10,7))
        ax = plt.axes()
        ax.spines['top'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.set_facecolor('white')
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        ax.tick_params( colors='black', labelsize='large', bottom=False)
        
        plt.ylabel('Seconds / Epoch', size=16)
        plt.xlabel('Batch Size', size=16)
        plt.title('Convolutional Model', color='black', size=26)



        x = [int(b) for b in self.batch_sizes]
        if type(gpus) == int:
            y = self.get_gpu(gpus)
            plt.plot(x, y, marker='o', color='red', markersize=12)
        elif type(gpus) == list:
            colors = ['red','yellow','green','blue']
            for i in range(len(gpus)):
                y = self.get_gpu(i)
                plt.plot(x, y, marker='o', color=colors[i], markersize=12, alpha=.6)
        ax.legend(['1 GPU', '2 GPUs', '3 GPUs', '4 GPUs'])


        _, top = plt.ylim()
        print(top)
        plt.ylim((0,top))

        plt.show()
            
    def batch_bars(self, batch_size):
        
        # styling
        plt.figure(facecolor='white', edgecolor='black', figsize=(8,10))
        ax = plt.axes()
        ax.set_facecolor('white')
        ax.spines['top'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        ax.tick_params( colors='black', labelsize='large', bottom=False)
        plt.ylabel('Seconds/Epoch', size=22)
        plt.title('Batch Size: {}'.format(batch_size), color='black', size=16)

                
        plt.xticks([1,2,3,4], ('1 GPU','2 GPUs','3 GPUs','4 GPUs'))
        times = self.get_batch(batch_size)
        plt.bar([1,2,3,4], times)
        plt.ylim((0,4))
        plt.show()



if __name__ == '__main__':
    #c = Grapher('linear.json')
    c = Grapher('conv.json')
    #for i in [20, 50, 100, 300]:
    #    c.batch_bars(i)
    c.plot_gpus([1,2,3,4])
        