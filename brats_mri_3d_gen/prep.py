# Download the toy dataset from MONAI
print("Downloadning BraTS2016/17")
from monai.apps import DecathlonDataset
from generative.losses.perceptual import PerceptualLoss


_ = DecathlonDataset(root_dir="/mnt/Datasets/Open-Datasets/MONAI", task="Task01_BrainTumour", section="training", download=True, cache_rate = 0.0)
# _ = DecathlonDataset(root_dir="/mnt/.node1/Open-Datsets/MONAI", task="Task01_BrainTumour", section="training", download=True)
perceptual_loss = PerceptualLoss(
    spatial_dims=3, 
    network_type="squeeze", 
    is_fake_3d=True, 
    fake_3d_ratio=0.2
)

# # Download the bigger dataset from Synapse
# print("Downloadning BraTS2023")
# import synapseclient
# syn = synapseclient.Synapse()
# syn.login('adam_peaston','AXXXXXXXXX2')
# syn51514132 = syn.get(entity='syn51514132', downloadFile=True, downloadLocation="/mnt/Datasets/strongcompute_adam/MONAI", ifcollision="overwrite.local")
# filepath = syn51514132.path
# print(f"BraTS2023-GLI downloaded to {filepath}")