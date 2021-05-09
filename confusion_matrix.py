import torch
import torch.nn as nn


# def confusion_matrix(model,dataloader,device):
#     nb_classes = 3

#     confusion_matrix = torch.zeros(nb_classes, nb_classes)
#     with torch.no_grad():
#         for i, (images, labels) in enumerate(dataloader):
#             images = images.to(device)
#             labels = labels.to(device)
#             outputs = model(images)
#             _, preds = torch.max(outputs, 1)
#             for t, p in zip(labels.view(-1), preds.view(-1)):
#                     confusion_matrix[t.long(), p.long()] += 1

#     print(confusion_matrix)
#     print(confusion_matrix.diag()/confusion_matrix.sum(1))

from sklearn.metrics import confusion_matrix, accuracy_score

def calculate_confusion_matrix(model,dataloader,device):
    nb_classes = 3

    # Initialize the prediction and label lists(tensors)
    predlist=torch.zeros(0,dtype=torch.long, device='cpu')
    lbllist=torch.zeros(0,dtype=torch.long, device='cpu')

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            # Append batch prediction results
            predlist=torch.cat([predlist,preds.view(-1).cpu()])
            lbllist=torch.cat([lbllist,labels.view(-1).cpu()])

    # Confusion matrix
    conf_mat=confusion_matrix(lbllist.numpy(), predlist.numpy())
    print(conf_mat)

    # Per-class accuracy
    class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)
    print("Class-wise accuracy=",class_accuracy)

    accuracy=accuracy_score(lbllist.numpy(), predlist.numpy())
    print("Accuracy=",accuracy)