
import functools
import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data.aligned_conc_dataset import AlignedConcDataset
from data.aligned_conc_dataset_noised import AlignedConcDatasetNoised
from data.dataset import AddGaussianNoise, AddSaltPepperNoise




def get_GaussianNoisetransforms(args):
    return transforms.Compose(

             [
            transforms.Resize((args.FINE_SIZE, args.FINE_SIZE)),
            transforms.RandomApply([AddGaussianNoise(mean=0, variance=args.noise)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = [0.4951, 0.3601, 0.4587],
                std = [0.1474, 0.1950, 0.1646],
            ),
        ]
    )

def get_SaltNoisetransforms(args):

        return transforms.Compose(

             [
            transforms.Resize((args.FINE_SIZE, args.FINE_SIZE)),
            transforms.RandomApply([AddSaltPepperNoise(density=0.1, p=args.noise/10)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = [0.4951, 0.3601, 0.4587],
                std = [0.1474, 0.1950, 0.1646],
            ),
        ]
    )




def get_data_loaders(args):
   
    base_transforms = transforms.Compose(
        [
            transforms.Resize((args.LOAD_SIZE, args.LOAD_SIZE)),
            transforms.RandomCrop((args.FINE_SIZE, args.FINE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
               mean = [0.4951, 0.3601, 0.4587],
                std = [0.1474, 0.1950, 0.1646],
            ),
        ]
    )


    train = AlignedConcDataset(args, data_dir=os.path.join(args.data_path, 'train'),transform=base_transforms)

    train_loader = DataLoader(
        train,
        batch_size=args.batch_sz,
        shuffle=True,
        num_workers=args.n_workers)



    if args.noise>0.0:
        if args.noise_type=='Gaussian':
            print('Gaussian')
            test_transforms=get_GaussianNoisetransforms(args)

        elif args.noise_type=='Salt':
            print("Salt")
            test_transforms = get_SaltNoisetransforms(args)

        test = AlignedConcDatasetNoised(args, data_dir=os.path.join(args.data_path, 'test'),rgb_transform=test_transforms,
            depth_transform = test_transforms)
        
        test_loader = DataLoader(
        test,
        batch_size=args.batch_sz,
        shuffle=False,
        num_workers=args.n_workers)
    else:
        test_transforms = transforms.Compose(
        [
            transforms.Resize((args.FINE_SIZE, args.FINE_SIZE)),

            transforms.ToTensor(),
            transforms.Normalize(
                mean = [0.4951, 0.3601, 0.4587],
                std = [0.1474, 0.1950, 0.1646],
            ),
        ])

        test = AlignedConcDataset(args, data_dir=os.path.join(args.data_path, 'test'),transform=test_transforms)

        test_loader = DataLoader(
        test,
        batch_size=args.batch_sz,
        shuffle=False,
        num_workers=args.n_workers)


    return train_loader, test_loader


     