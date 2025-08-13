import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score,confusion_matrix
import numpy as np
import torch.nn.functional as F
import pandas as pd
from collections import defaultdict
import os



def calculate_metrics(all_predictions,all_labels):
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='macro',zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')
    kappa = cohen_kappa_score(all_labels, all_predictions)
    qwk = cohen_kappa_score(all_labels, all_predictions,weights='quadratic')
    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, \
    F1 Score: {f1}, Cohen's Kappa: {kappa}, Quadratic Weighted Kappa: {qwk}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'kappa': kappa,
        'qwk': qwk
    }


def compute_all_metrics(outputs, labels,binary_list,multi_list):
    metrics = {} 
    if multi_list:      #DR grading test
        DR_probs= torch.softmax(outputs[:,-5:], dim=1)   
        pred = torch.argmax(DR_probs,dim=1)
        labels_item = labels[:, -1].float()
        metrics[f'multi_{0}'] = calculate_metrics(pred.cpu().numpy(), labels_item.cpu().numpy())

    # binary task test
    outputs[:,0:-5] = torch.sigmoid(outputs[:,0:-5])
    binary_pred = (outputs[:,0:-5]> 0.5).float()
 
    for i in range(len(binary_list)):
        pred=binary_pred[:,binary_list[i]]
        labels_item = labels[:, binary_list[i]].float()
        metrics[f'binary_{binary_list[i]}'] = calculate_metrics(pred.cpu().numpy(), labels_item.cpu().numpy())
    return metrics



# calculate average metrics cross task and datasets
def calculate_average_metrics(all_metrics):   
    if not all_metrics:
        return pd.DataFrame()
    combied_metrics=pd.concat(all_metrics,ignore_index=False) 
    combied_metrics_mean = combied_metrics.groupby(combied_metrics.index).mean() 
    all_metrics_mean = combied_metrics_mean.mean()

    combied_metrics_mean.loc['total_average'] =all_metrics_mean
    return combied_metrics_mean

def write_metrics_to_csv(zip_metrics, filename):
    if os.path.exists(filename):
        zip_metrics.to_csv(filename, mode='a',header=False,index=True)
    else:
        zip_metrics.to_csv(filename, mode='w',header=True,index=True)



def evaluate(dataloader, net, args,if_valid,if_final_test):
    all_metrics_seen = []
    all_metrics_unseen = []
    all_metrics_ODIR= []
    all_metrics=[]
    Softmax = torch.nn.Softmax(dim=1)
    net.eval()
    with torch.no_grad():
        for dataset_name, dataset_info in dataloader.items():    # Process the test and validation datasets sequentially, one by one.
            all_output=[]
            all_label=[]
            print(f"Processing {dataset_name}")
            dataloader = dataset_info['dataloader']
            binary_list = dataset_info['binary_list']
            multi_list = dataset_info['multi_list']
            for i,data in enumerate(dataloader):
                if torch.cuda.is_available():
                    device = torch.device("cuda")
                else:
                    device = torch.device("cpu")
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = net(inputs) 
        
                all_output.append(outputs)
                all_label.append(labels)
         
            all_output = torch.cat(all_output)
            print(all_output.shape)
            all_label = torch.cat(all_label)
            print(all_label.shape)
            metrics=compute_all_metrics(all_output, all_label, binary_list,multi_list)
            metrics_df=pd.DataFrame.from_dict(metrics, orient='index')
            if not if_valid:
                if dataset_name in ['ADAM_testset','PALM_testset','DDR_testset','HR_testset','Cataract_testset','REFUGE_testset']:
                    all_metrics_seen.append(metrics_df)
                elif dataset_name in ['APTOS_testset','HPMI_testset','ORIGA_testset','RFMiD_testset']:
                    all_metrics_unseen.append(metrics_df)
                else:
                    all_metrics_ODIR.append(metrics_df)
            else:
                all_metrics_seen.append(metrics_df)
            all_metrics.append(metrics_df)
        all_metric_average_seen = calculate_average_metrics(all_metrics_seen)   # average metric on meta-dataset
        all_metric_average_unseen = calculate_average_metrics(all_metrics_unseen) # average metric on unseen-dataset
        all_metric_average_ODIR = calculate_average_metrics(all_metrics_ODIR) # average metric on ODIR dataset
        all_metrics_df=pd.concat(all_metrics,ignore_index=False)

        if if_final_test:
            outputPath = os.path.join('./exp/test_result/', args.post)
            if not os.path.exists(outputPath):  
                os.makedirs(outputPath)
            write_metrics_to_csv(all_metrics_df,outputPath+'/all_dataset_result.csv')
            write_metrics_to_csv(all_metric_average_seen,outputPath+'/seen_dataset_result.csv')
            write_metrics_to_csv(all_metric_average_unseen,outputPath+'/unseen_dataset_result.csv')
            write_metrics_to_csv(all_metric_average_ODIR,outputPath+'/ODIR_dataset_result.csv')
            return 
        else:
            if if_valid:
                return all_metric_average_seen.loc['total_average']
            else:
                return all_metric_average_seen.loc['total_average'],all_metric_average_unseen.loc['total_average'],all_metric_average_ODIR.loc['total_average']
            
