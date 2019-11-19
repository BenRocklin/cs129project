import image_reader
import models

def main():
    features, labels = image_reader.getTrainableDataset()
    models.trainModels(features, labels)

if __name__ == "__main__":
    main()