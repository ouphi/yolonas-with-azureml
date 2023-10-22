import sys

from super_gradients.training import Trainer, models, dataloaders, KDTrainer
from super_gradients.training import dataloaders, models
from torchvision import transforms
from super_gradients.training import training_hyperparams
from super_gradients.training.losses import KDLogitsLoss, LabelSmoothingCrossEntropyLoss


experiment_name = "kdtraining"
checkpoint_dir = sys.argv[1]

# Create Knowledge Distillation Trainer
kd_trainer = KDTrainer(experiment_name=experiment_name, ckpt_root_dir=checkpoint_dir)

# Load teacher model
pretrained_beit = models.get('beit_base_patch16_224', arch_params={'num_classes': 10, "image_size": [224, 224], "patch_size": [16, 16]}, pretrained_weights="cifar10")

# Load student model
student_resnet18 = models.get('resnet18_cifar', num_classes=10)

# Load dataset
train_dataloader = dataloaders.get("cifar10_train", dataloader_params={"batch_size": 128})
val_dataloader = dataloaders.get("cifar10_val", dataloader_params={"batch_size": 512})

# Setup training hyperparameters
kd_params = {
    "max_epochs": 3,          # We only train for 3 epochs because this is just an example
    'lr_cooldown_epochs': 0,  # We dont want to use lr cooldown since we only train for 3 epochs
    'lr_warmup_epochs': 0,    # We dont want to use lr  warmup  since we only train for 3 epochs
    "loss": KDLogitsLoss(distillation_loss_coeff=0.8, task_loss_fn=LabelSmoothingCrossEntropyLoss()),
    "loss_logging_items_names": ["Loss", "Task Loss", "Distillation Loss"]}

training_params = training_hyperparams.get("imagenet_resnet50_kd", overriding_params=kd_params)
arch_params={"teacher_input_adapter": transforms.Resize(224)}

# Train model
kd_trainer.train(training_params=training_params,
                 student=student_resnet18,
                 teacher=pretrained_beit,
                 kd_architecture="kd_module",
                 kd_arch_params=arch_params,
                 train_loader=train_dataloader, valid_loader=val_dataloader)
