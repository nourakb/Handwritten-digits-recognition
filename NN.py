import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from skimage.util import view_as_blocks



transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

# Récupérer les données d'apprentissage et de test
train_dataset = datasets.MNIST(root='./data', train=True,download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False,download=True, transform=transform)


#fonction qui divise une image en 16 blocs
def slice(image):
    block_shape = (7, 7)
    cells= view_as_blocks(image.numpy(), block_shape)
    flatten_cells = cells.reshape((16,7,7))
    return flatten_cells


new_train_data=[] 
new_test_data=[]

#On crée une nouvel dataset qui contient les images divisés en 16 blocs 
for image in train_dataset.data:
    x=slice(image)
    new_train_data.append(x)
    
for image in test_dataset.data:
    x=slice(image)
    new_test_data.append(x)


#fonction qui extracte les 3 fonctionnalités f1 ,f2 ,f3
def extract_features(image):
    extracted_image= np.array([[0,0,0]])
    N=np.count_nonzero(image)
    for cell in image:
        #f1
        n=np.count_nonzero(cell)
        f1=n/N
        #f2 et f3
        X= np.nonzero(cell)[1].reshape(-1, 1)
        Y= np.nonzero(cell)[0]
        if(X.size != 0):
            linear_regressor = LinearRegression()  # create object for the class
            linear_regressor.fit(X, Y)  # perform linear regression
            b=linear_regressor.coef_[0]
            
            f2=(2*b)/(1+pow(b,2))
            f3=(1-pow(b,2))/(1+pow(b,2))
        else : 
            f2=0
            f3=0
       
        features=[f1,f2,f3] 
        
        extracted_image=np.append(extracted_image,[features], axis=0)
    extracted_image=np.delete(extracted_image,0,axis=0)
    return extracted_image

#Chaque image aura 16 blocs chaqun va contenir 3 fonctionnalités (3 float)
for i in range(len(new_test_data)):
    new_test_data[i]=extract_features(new_test_data[i])
for i in range(len(new_train_data)):
    new_train_data[i]=extract_features(new_train_data[i])

#dataset d'apprentissage=50 000 échantillons , dataset de validation = 10 000 échantillons
train_data,validation_data,train_target,validation_target=train_test_split(new_train_data,train_dataset.targets,test_size=10000)
test_data=new_test_data
test_target=test_dataset.targets

#Convertire les images en tensors
for i in range(len(train_data)):
    train_data[i]=torch.from_numpy(train_data[i])
for i in range(len(test_data)):
    test_data[i]=torch.from_numpy(test_data[i])
for i in range(len(validation_data)):
    validation_data[i]=torch.from_numpy(validation_data[i])

class MyDataset(Dataset):
    def __init__(self,data,targets):
        self.data=data
        self.targets=targets
    def __getitem__(self,idx):
        return (self.data[idx],self.targets[idx])
    def __len__(self):  
        return len(self.targets)   


my_training_dataset=MyDataset(train_data,train_target)
my_test_dataset=MyDataset(test_data,test_target)
my_validation_dataset=MyDataset(validation_data,validation_target)   

batch_size=64
train_loader=torch.utils.data.DataLoader(my_training_dataset,batch_size=batch_size,shuffle=True)
test_loader=torch.utils.data.DataLoader(my_test_dataset,batch_size=batch_size,shuffle=True)
validation_loader=torch.utils.data.DataLoader(my_validation_dataset,batch_size=batch_size,shuffle=True)

#creation de notre modele
model = nn.Sequential(nn.Linear(48,80),
                      nn.ReLU(),
                      nn.Linear(80,80),
                      nn.ReLU(),
                      nn.Linear(80, 10),
                      )

# Définir la fonction du coût. On peut choisir CrossEntropyLoss
loss=nn.CrossEntropyLoss()
# Définir une fonction d'optimisation des coût: Adam par exemple. On devra définir un learning rate. On choisira 0.001.
opt = optim.Adam(model.parameters(), lr=0.001)
# Définir le nombre d'epochs. Commencer petit. 
n_epochs = 15


#########Boucle d'entrainement et de validation###########
# Créer une boucle sur les epochs:
for epoch in range(n_epochs):
    # Spécifier qu'on est sur le mode entraînement
    model.train()
    # initialiser notre coût d'apprentissage à 0.0
    t_cost= 0.0
    # Boucler sur les minibatchs des données d'entaînement (les données et leurs targets):
    for batch in train_loader:
        # le vecteur des labels prédites par le modèle est le résultat de l'application du modèle sur le minibatch en cours. 
        # Nous aurons besoin d'applatir les données avant de les donner à notre NN
        inputs , labels = batch
        inputs = inputs.float() 
        labels = labels.float()
        outputs = model(torch.flatten(inputs,start_dim=1))
        # Calculer le coût en comparant les labels prédits aux targets du minibatch
        cout = loss(outputs,labels.long())
        # Backpropagation: 
        # Réinitialiser l'optimiseur
        opt.zero_grad()
        # Faire la backpropagation
        cout.backward()
        # Effectuer un pas d'optimisation
        opt.step()
        # Mettre à jour votre coût d'apprentissage en lui ajoutant le coût du data batch
        t_cost += cout.item()
        # A la sortie de la boucle de l'entraînement, on calcule le coût moyen pour toutes les données training
        t_cost /= (len(train_loader))


    # 
    v_cost= 0.0
    correct=0
    total=0
    model.eval()
    # Indiquer à Pytorch qu'on ne va pas faire de Gradient descent (comme on est dans l'évaluation)
    for batch in validation_loader: 
        inputs, labels = batch
        inputs = inputs.float()
        labels = labels.float()
        with torch.no_grad():
            outputs = model(torch.flatten(inputs,start_dim=1))
       
            
            
        cout = loss(outputs,labels.long())#lach long too
        
        v_cost += cout.item()   
        correct += torch.sum(torch.argmax(outputs, dim=1)== labels).item() 
    # A la sortie de cette boucle, calculer le coût moyen de validation
    v_cost = v_cost/(len(validation_loader))
    # Calculer la précision: la moyenne des prévisions correctes sur l'ensemble des observations dans le dataset validation 
    correct /= len(validation_loader.dataset)
    # Afficher pour chaque itération le coût d'entraînement, le coût de validation, et la précision.
    #print("coût d'entraînement :",t_cout_moy)
    print(f"epoch: {epoch+1}, train loss: {t_cost:.6f}, validation loss: {v_cost:.4f} , correct prediction: {correct*100:.2f} ")
torch.save(model,'mymodel') 



#########Boucle de test###########

with torch.no_grad(): #After we finshed training we don't update the grad anymore
    correct_preds = 0
    total_preds = 0
    for images, labels in test_loader:
        
        images = images.float()
        labels = labels.float()
        
        output = model(torch.flatten(images,start_dim=1))
        
        _, preds = torch.max(output.data, 1)
        
        total_preds += labels.size(0)
        correct_preds += (preds == labels).sum().item()
    
    print(f'Accuracy: {100 * correct_preds / total_preds}%')
torch.save(model,'mymodel')     
import random

#Visualiser les résultats de la reconnaissance de 8 images

n_images = 8

with torch.no_grad():
    for i in range(n_images):
        plt.subplot(2, n_images // 2, i + 1)
        index = random.randint(0, total_preds - 1)
        img, label =my_test_dataset[index]
        img=img.float()
        label=label.float()
        temp_img = img.reshape(-1, 1*48)
        output = model(temp_img)
        _, pred = torch.max(output.data, 1)
        
        image=test_dataset.data[index]
        plt.imshow(image.squeeze(0).numpy())
        plt.title(f'Prediction: {pred.item()}\nActual: {label}')
        plt.axis('off')
    
    plt.show()
