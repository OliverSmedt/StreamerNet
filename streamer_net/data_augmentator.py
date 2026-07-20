from torchvision import transforms


#Definition for a transform which randomly scales, moves, translates and discolors images
augmentation_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.ColorJitter(0.5, 0.5,0.5 ) ,
                                             transforms.RandomAffine(45, (0,0.3), (0.5, 1.5))])