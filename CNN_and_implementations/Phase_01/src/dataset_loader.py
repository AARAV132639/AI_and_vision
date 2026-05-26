from torchvision.datasets import FashionMNIST

def download_dataset():
    dataset= FashionMNIST(
        root= "./datasets",
    train= True,
    download= True)

    test_dataset= FashionMNIST(
    root= "./datasets",
    train= False,
    download= True)
    return dataset, test_dataset

if __name__=="__main__":
    train_data, test_data= download_dataset()
    print("Dataset downloaded successfully")
    print("Training samples:", len(train_data))
    print("Test samples:", len(test_data))
