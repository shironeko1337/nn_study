global last_accuracy
last_accuracy= 0

# evaluate function using raw test data, normalizer and comparator
# return should break or not by the compare result
def evaluate(network, raw_samples,raw_labels,normalize_data = None, compare_result = None):
    correct = 0
    for i,raw_sample in enumerate(raw_samples):
      # here we don't need the normalized label since it's for calculating loss
      sample,label = normalize_data(raw_sample,raw_labels[i])
      network.set_sample(sample)
      network.forward()
      correct += compare_result(network.layers[-1].output,raw_labels[i])
    accuracy = float(correct) / float(len(raw_samples))

    global last_accuracy
    if last_accuracy > accuracy:
       return True
    last_accuracy = accuracy
    print('Accuracy:', accuracy)
    return False
