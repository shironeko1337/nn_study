from dataset.reader import load_mnist
from nn import Network
from evaluator import evaluate

import numpy as np

if __name__ == '__main__':
    # x: [[array(...28)*28] * 600000]
    # y: [int * 60000]
    (x_train, y_train), (x_test, y_test) = load_mnist()

    def compare_result(output,raw_label):
       return raw_label == sorted([(-output[i],i) for i in range(10)])[0][1]

    def normalize_data(raw_sample,raw_label):
      sample = np.array([c for r in raw_sample for c in r]).reshape(-1, 1) # r = row, c = column
      sample = sample/256
      label = np.array([0] * 10).reshape(-1, 1)
      label[raw_label] = 1
      return sample,label

    # initialization
    nn = Network([784,300,300,10])
    epoch = 100
    learning_rate = 0.1
    for _ in range(epoch):
      for i,raw_sample in enumerate(x_train):
        # normalize sample and label
        sample,label = normalize_data(raw_sample,y_train[i])

        nn.train_once(
          sample = sample,
          label = label,
          learning_rate = learning_rate
        )

      if epoch % 10 == 0:
         should_break = evaluate(nn,x_test,y_test,normalize_data = normalize_data, compare_result = compare_result)
         if should_break:
            break





