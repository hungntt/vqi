import torch.distributed.algorithms.model_averaging.averagers
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet101, deeplabv3_resnet50

from model.segmentation.segmentation_output_wrapper import SegmentationModelOutputWrapper


class XAIEvaluation:
    def __init__(self, model, data):
        if model == 'resnet101':
            self.model = deeplabv3_resnet101(pretrained=True, progress=False)
        elif model == 'resnet50':
            self.model = deeplabv3_resnet50(pretrained=True, progress=False)

        self.model = self.model.eval()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.model = SegmentationModelOutputWrapper(self.model)

        self.data = data

    def dataloader(self):
        # Load datasets and make loaders.
        test_samples = 24
        transformer = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        train_set = torchvision.datasets.MNIST(root='./sample_data', train=True, transform=transformer, download=True)
        test_set = torchvision.datasets.MNIST(root='./sample_data', train=False, transform=transformer, download=True)
        train_loader = DataLoader(train_set, batch_size=200, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=200, pin_memory=True)

        # Load a batch of inputs and outputs to use for evaluation.
        x_batch, y_batch = iter(test_loader).next()
        x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

    def evaluate(self):
        logits = torch.Tensor().to(self.device)
        targets = torch.LongTensor().to(self.device)

        with torch.no_grad():
            for images, labels in self.data:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)

                logits = torch.cat((logits, outputs['out']), 0)
                targets = torch.cat((targets, labels), 0)





