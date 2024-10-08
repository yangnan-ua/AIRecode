## This is a study on Imbalanced CIFAR10 dataset, mainly for discover the reweighting loss's function on the imbanlaced dataset.
### First, we need to get an imbalanced dataset.


Since CIFAR10 is balance, so we need to do some processing on it. [imbalance_cifar.py](https://github.com/yangnan-ua/AIRecode/blob/main/imbalance_cifar.py)
![image](https://github.com/user-attachments/assets/58a2d569-88c9-4586-91c7-deab95371202)

---
### Second, train it by ResNet18 with Origin Loss and Reweighting Loss
![image](https://github.com/user-attachments/assets/03b6af5e-9b74-408d-acb3-20191a3ab21d)


![image](https://github.com/user-attachments/assets/9cd62042-d3f4-4e63-b3d2-edf5bb4d17c9)



The wight is $\frac{Num Classes \times Class Count}{Total Samples Num}$ 


And then use this weight on the CrossEntropyLoss, can get the reweighting loss.
```python
def compute_class_weights(loader):
    class_counts = Counter()
    for _, targets in loader:
        class_counts.update(targets.cpu().numpy())
    total_samples = sum(class_counts.values())
    class_weights = {cls: total_samples / (len(class_counts) * count) for cls, count in class_counts.items()}
    return class_weights

class_weights = compute_class_weights(train_loader)
weights = torch.FloatTensor([class_weights[i] for i in range(len(class_weights))]).to(device)
criterion = nn.CrossEntropyLoss(weight=weights).to(device)
```

---
## Conclusion: 
From the experiment, we can know that the shortcoming of reweighting loss with $weights = \frac{C}{Class Count}$  (C means constant)is that it will **ignore the frequency class**. So it's better to use other loss to concentrate on both frequency and minor classes.

## Note:
My code is mainly based on [kaidic](https://github.com/kaidic/LDAM-DRW) and [ZOMIN28](https://github.com/ZOMIN28/ResNet18_Cifar10_95.46).
