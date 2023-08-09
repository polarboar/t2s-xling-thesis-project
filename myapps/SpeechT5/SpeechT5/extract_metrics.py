def get_epoch(line):
    start_idx = line.find('"epoch"') + len('"epoch": ')
    end_idx = line.find(',', start_idx)
    epoch = line[start_idx:end_idx]
    return int(epoch)
    print(f'Epoch: {epoch}')
    
def get_metric(metric_name, line):
    metric_name = '"' + metric_name + '"'
    start_idx = line.find(metric_name) + len(metric_name) + 3
    end_idx = line.find('"', start_idx)
    metric = line[start_idx:end_idx]
    #print(f'Metric Name: {metric_name}, Value: {metric}')
    return float(metric)

def print_metric(metrics_list, metric_name):
    for metrics in metrics_list:
        print(metrics[metric_name])

f = open('./slurm-1653134.out')
train_metrics = []
valid_metrics = []

train_metrics_to_get = ['train_s2c_loss', 'train_loss', 'train_s2c_accuracy']
valid_metrics_to_get = ['valid_s2c_loss', 'valid_loss', 'valid_s2c_accuracy']
for l in f:
    
    # Get Train metrics
    if 'INFO | train |' in l:
        metrics = {}
        #print(l)
        epoch = get_epoch(l)
        metrics['epoch'] = epoch
        for metric_name in train_metrics_to_get:
            metric = get_metric(metric_name, l)
            metrics[metric_name] = metric
        #print(metrics)
        train_metrics.append(metrics)
        
    # Get Validation metrics
    if 'INFO | valid |' in l:
        metrics = {}
        #print(l)
        epoch = get_epoch(l)
        metrics['epoch'] = epoch
        for metric_name in valid_metrics_to_get:
            metric = get_metric(metric_name, l)
            metrics[metric_name] = metric
        #print(metrics)
        valid_metrics.append(metrics)

        
#print(train_metrics)
#print(valid_metrics)

print('Printing Epoch Numbers')
print_metric(train_metrics, 'epoch')
print(f'Printing train_loss')
print_metric(train_metrics, 'train_loss')
 
print('Printing Epoch Numbers')
print_metric(valid_metrics, 'epoch')
print(f'Printing valid_loss')
print_metric(valid_metrics, 'valid_loss')
    
