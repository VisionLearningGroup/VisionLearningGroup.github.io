import torch.utils.data
import torchnet as tnt
from builtins import object
import torchvision.transforms as transforms
from datasets_ import Dataset


class PairedData(object):
    def __init__(self, data_loader_A, data_loader_B, data_loader_C, data_loader_D, data_loader_t, max_dataset_size):
        self.data_loader_A = data_loader_A
        self.data_loader_B = data_loader_B
        self.data_loader_C = data_loader_C
        self.data_loader_D = data_loader_D
        self.data_loader_t = data_loader_t
     
        self.stop_A = False
        self.stop_B = False
        self.stop_C = False
        self.stop_D = False
        self.stop_t = False
        self.max_dataset_size = max_dataset_size

    def __iter__(self):
        self.stop_A = False
        self.stop_B = False
        self.stop_C = False
        self.stop_D = False
        self.stop_t = False

        self.data_loader_A_iter = iter(self.data_loader_A)
        self.data_loader_B_iter = iter(self.data_loader_B)
        self.data_loader_C_iter = iter(self.data_loader_C)
        self.data_loader_D_iter = iter(self.data_loader_D)
        self.data_loader_t_iter = iter(self.data_loader_t)
        self.iter = 0
        return self

    def __next__(self):
        A, A_paths = None, None
        B, B_paths = None, None
        C, C_paths = None, None
        D, D_paths = None, None
        t, t_paths = None, None
        try:
            A, A_paths = next(self.data_loader_A_iter)
        except StopIteration:
            if A is None or A_paths is None:
                self.stop_A = True
                self.data_loader_A_iter = iter(self.data_loader_A)
                A, A_paths = next(self.data_loader_A_iter)

        try:
            B, B_paths = next(self.data_loader_B_iter)
        except StopIteration:
            if B is None or B_paths is None:
                self.stop_B = True
                self.data_loader_B_iter = iter(self.data_loader_B)
                B, B_paths = next(self.data_loader_B_iter)
        try:
            C, C_paths = next(self.data_loader_C_iter)
        except StopIteration:
            if C is None or C_paths is None:
                self.stop_C = True
                self.data_loader_C_iter = iter(self.data_loader_C)
                C, C_paths = next(self.data_loader_C_iter)
        try:
            D, D_paths = next(self.data_loader_D_iter)
        except StopIteration:
            if D is None or D_paths is None:
                self.stop_D = True
                self.data_loader_D_iter = iter(self.data_loader_D)
                D, D_paths = next(self.data_loader_D_iter)

        try:
            t, t_paths = next(self.data_loader_t_iter)
        except StopIteration:
            if t is None or t_paths is None:
                self.stop_t = True
                self.data_loader_t_iter = iter(self.data_loader_t)
                t, t_paths = next(self.data_loader_t_iter)

        if (self.stop_A and self.stop_B and self.stop_C and self.stop_D and self.stop_t) or self.iter > self.max_dataset_size:
            self.stop_A = False
            self.stop_B = False
            self.stop_C = False
            self.stop_D = False
            self.stop_t = False
            raise StopIteration()
        else:
            self.iter += 1
            return {'S1': A, 'S1_label': A_paths,
                    'S2': B, 'S2_label': B_paths,
                    'S3': C, 'S3_label': C_paths,
                    'S4': D, 'S4_label': D_paths,
                    'T': t, 'T_label': t_paths}


class UnalignedDataLoader():
    def initialize(self, source, target, batch_size1, batch_size2, scale=32):
        transform = transforms.Compose([
            transforms.Resize(scale),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        #dataset_source1 = Dataset(source[1]['imgs'], source['labels'], transform=transform)
        dataset_source1 = Dataset(source[0]['imgs'], source[0]['labels'], transform=transform)
        data_loader_s1 = torch.utils.data.DataLoader(dataset_source1, batch_size=batch_size1, shuffle=True, num_workers=4)
        self.dataset_s1 = dataset_source1

        dataset_source2 = Dataset(source[1]['imgs'], source[1]['labels'], transform=transform)
        data_loader_s2 = torch.utils.data.DataLoader(dataset_source2, batch_size=batch_size1, shuffle=True, num_workers=4)
        self.dataset_s2 = dataset_source2

        dataset_source3 = Dataset(source[2]['imgs'], source[2]['labels'], transform=transform)
        data_loader_s3 = torch.utils.data.DataLoader(dataset_source3, batch_size=batch_size1, shuffle=True, num_workers=4)  
        self.dataset_s3 = dataset_source3      

        dataset_source4 = Dataset(source[3]['imgs'], source[3]['labels'], transform=transform)
        data_loader_s4 = torch.utils.data.DataLoader(dataset_source4, batch_size=batch_size1, shuffle=True, num_workers=4)  
        self.dataset_s4 = dataset_source4     

        #for i in range(len(source)):
        #    dataset_source[i] = Dataset(source[i]['imgs'], source[i]['labels'], transform=transform)
        dataset_target = Dataset(target['imgs'], target['labels'], transform=transform)
        data_loader_t = torch.utils.data.DataLoader(dataset_target, batch_size=batch_size2, shuffle=True, num_workers=4)
        

        self.dataset_t = dataset_target
        self.paired_data = PairedData(data_loader_s1, data_loader_s2, data_loader_s3,data_loader_s4, data_loader_t,
                                      float("inf"))
       

    def name(self):
        return 'UnalignedDataLoader'

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return min(max(len(self.dataset_s1),len(self.dataset_s2),len(self.dataset_s3), len(self.dataset_s4),len(self.dataset_t)), float("inf"))
