from train_conv import train_conv
from train_linear import train_linear
import json

def get_stats(training_function, epochs:int, dev_ids:list, batch_sizes:list):

    stats = {}
    print(training_function, dev_ids, batch_sizes)
    for i in range(len(dev_ids)):
        cur_dev_ids = dev_ids[:i+1]
        print('\tCur Devs:', cur_dev_ids)
        stats[len(cur_dev_ids)] = {}
        for batch_size in batch_sizes:
            print('\t\tBatch Size:', batch_size)

            stats[len(cur_dev_ids)][batch_size] = training_function(epochs, batch_size, cur_dev_ids)
            print(stats[len(cur_dev_ids)][batch_size])

    return stats

epochs = 20
dev_ids = [0,1,2,3]
batch_sizes = [10,20,30,50,70,100,150,200,300,400,500,600]

if __name__ == '__main__':
    linear_stats = get_stats(train_linear, epochs=epochs, dev_ids=dev_ids, batch_sizes=batch_sizes)
    conv_stats = get_stats(train_conv, epochs=epochs, dev_ids=dev_ids, batch_sizes=batch_sizes)

    print(linear_stats)
    linear_json = json.dumps(linear_stats, indent=4)
    with open('linear.json', 'w') as f:
        f.write(linear_json)

    print(conv_stats)
    conv_json = json.dumps(conv_stats, indent=4)
    with open('conv.json', 'w') as f:
        f.write(conv_json)

