import argparse
import logging

from pathlib import Path
import torch
import torch.nn as nn
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils.data_loading_custom import BasicDataset
from evaluate import evaluate
from torch_ema import ExponentialMovingAverage

from nets.unet_zoo import AttU_Net
from nets.unet import UNet2Plus

dir_img = Path('./data/img_4CH/')
dir_mask = Path('./data/GT_4CH/')
dir_unlabeled_img=Path('./data/unlabeled_img_4CH')
dir_checkpoint = Path('./checkpoints/')


def train_net(net_student,
              net_teacher,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 1e-5,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False):
    # 1. Create dataset
    dataset = BasicDataset(dir_img, dir_mask,dir_unlabeled_img, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val

    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    # unlabeled_loader=DataLoader(unlabeled_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer_student = optim.RMSprop(net_student.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler_student = optim.lr_scheduler.ReduceLROnPlateau(optimizer_student, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    ema=ExponentialMovingAverage(net_student.parameters(),decay=0.995)
    ema.to(device=device)
    bceloss=nn.BCELoss()
    mseloss=nn.MSELoss()
    celoss=nn.CrossEntropyLoss()

    global_step = 0

    # 5. Begin training
    best_dice = 0.
    theta=0

    for epoch in range(1, epochs+1):
        net_student.train()
        net_teacher.train()
        epoch_loss = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='imgs') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']
                unlabeled_images=batch['unlabeled_img']

                assert images.shape[1] == net_student.n_channels, \
                    f'Network has been defined with {net_student.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                unlabeled_images = unlabeled_images.to(device=device, dtype=torch.float32)

                with torch.cuda.amp.autocast(enabled=amp):
                    teacher_y_1 = net_teacher(images)
                    teacher_y_2 = net_teacher(unlabeled_images)[-1]
                    if args.deepsupervision :
                        loss1=0
                        for i in range(len(teacher_y_1)):
                            loss1+=celoss(teacher_y_1[i],true_masks)*((i+1)/10)

                    else:
                        loss1 = bceloss(teacher_y_1, true_masks)

                    student_y_1 = net_student(unlabeled_images)[-1]
                    student_y_2 = net_student(unlabeled_images)[-1]
                    loss2 =mseloss(teacher_y_2,student_y_2)

                loss_total=0.5*loss2+0.5*loss1

                optimizer_student.zero_grad(set_to_none=True)
                grad_scaler.scale(loss_total).backward()
                grad_scaler.step(optimizer_student)
                grad_scaler.update()


                ema.update()
                ema.copy_to(net_teacher.parameters())
                theta=net_student.parameters()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss_total.item()
                experiment.log({
                    'train loss': loss_total.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss_total.item()})

                # Evaluation round
                division_step = (n_train // (10 * batch_size))

                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in net_student.named_parameters():
                            tag = tag.replace('/', '.')
                            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())

                        val_score = evaluate(net_student, val_loader, device,args.deepsupervision)
                        scheduler_student.step(val_score)

                        # f1=dict_mean(dict=class_stat['F1'],num_classes=args.classes)
                        # auc=dict_mean(dict=class_stat['AUC'],num_classes=args.classes)
                        # acc=dict_mean(dict=class_stat['ACC'],num_classes=args.classes)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        # logging.info('F1 score: {}'.format(class_stat['F1']))
                        # logging.info('auc score: {}'.format(class_stat['AUC']))
                        # logging.info('accuracy: {}'.format(class_stat['ACC']))

                        experiment.log({
                            'learning rate': optimizer_student.param_groups[0]['lr'],
                            'validation Dice': val_score,
                            # 'F1 score':f1,
                            # 'AUC score':auc,
                            # 'ACC score':acc,
                            'images': wandb.Image(images[0].cpu()),
                            'masks': {
                                'true': wandb.Image(true_masks[0].float().cpu()),
                                'pred': wandb.Image(torch.softmax(student_y_1, dim=1).argmax(dim=1)[0].float().cpu()),
                            },
                            'step': global_step,
                            'epoch': epoch,
                            **histograms
                        })
        if val_score > best_dice:
            best_dice = val_score
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net_student.state_dict(), "checkpoints_unet_custom/best_model_{}_epoch{}.pth".format(net_student.name, epoch))
            logging.info(f'best model {net_student.name} saved in {epoch}epoch!')
        # if save_checkpoint:
        #     Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
        #     torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
        #     logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=2, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--deepsupervision',type=bool,default=True,help='use deep supervision')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel

    # net = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    # net = UNet2Plus(deepsupervision=args.deepsupervision,n_channels=1,n_classes=args.classes)
    # net = UNet3Plus(n_channels=1,n_classes=args.classes)
    # net = UNet3Plus_DeepSup(n_channels=1,n_classes=args.classes)
    # net = UNet3Plus_DeepSup_CGM(n_channels=1,n_classes=args.classes)
    # net = DeepLabV3(num_classes=args.classes)
    # net = SSCFNet(n_classes=args.classes)
    net_student=UNet2Plus(deepsupervision=args.deepsupervision,n_channels=3,n_classes=args.classes)
    net_teacher=UNet2Plus(deepsupervision=args.deepsupervision,n_channels=3,n_classes=args.classes)
    logging.info(f'Network:\n'
                 f'\t{net_student.n_channels} input channels\n'
                 f'\t{net_student.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net_student.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net_student.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net_student.to(device=device)
    net_teacher.to(device=device)
    try:
        train_net(net_student=net_student,
                  net_teacher=net_teacher,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  amp=args.amp)
    except KeyboardInterrupt:
        torch.save(net_student.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise
