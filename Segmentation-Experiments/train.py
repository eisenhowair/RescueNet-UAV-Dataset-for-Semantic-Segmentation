# train.py
import os
import logging
import numpy as np
import time
import cv2
import argparse
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import transforms as ext_transforms
from tensorboardX import SummaryWriter
from util import transform, config
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU
from data.rescuenet import RescueNet as dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

## to implement Transformer
from models.factory import create_segmenter

from data.utils import median_freq_balancing, enet_weighing


def get_parser():
    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation")
    parser.add_argument(
        "--config",
        type=str,
        default="config/michael/rescuenet-pspnet101.yaml",
        help="config file",
    )
    parser.add_argument(
        "opts",
        help="see config/michael/rescuenet-pspnet101.yaml for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)

    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def calculate_class_weights(train_loader, num_classes):
    """
    Calculate class weights based on class frequencies in the dataset
    Moins bons résultats que median_freq ou enet
    """
    class_counts = torch.zeros(num_classes)
    print("Calculating class weights...")
    for _, target in train_loader:
        for c in range(num_classes):
            class_counts[c] += (target == c).sum()

    # Prevent division by zero
    class_counts = torch.where(
        class_counts > 0, class_counts, torch.ones_like(class_counts)
    )

    # Calculate weights using inverse frequency
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * num_classes

    # Optional: Normalize weights to sum to 1
    weights = weights / weights.sum()

    return weights.cuda()


def main_process():
    return not args.multiprocessing_distributed or (
        args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0
    )


def main():
    print("CUDA available:", torch.cuda.is_available())
    print("Current device:", torch.cuda.current_device())
    print("Device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("Device name:", torch.cuda.get_device_name(0))
    args = get_parser()

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in args.train_gpu)

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False

    main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    """initialisation des variables parce que sinon ça bugge, même si elles sont
    jamais utilisées"""
    train_epochs = []
    train_loss = []
    train_accuracy = []

    val_epochs = []
    val_loss = []
    val_accuracy = []
    global args
    global logger, writer
    logger = get_logger()

    args = argss
    # print(args)

    BatchNorm = nn.BatchNorm2d
    params_list = []

    base_transforms = A.Compose(
        [
            A.Resize(height=args.train_h, width=args.train_w),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        additional_targets={"mask": "mask"},
    )

    train_data = dataset(args.data_root, mode="train", transforms=base_transforms)
    print(
        f"Image de dimension {args.train_h} pour height et {args.train_w} pour width "
    )

    if args.transformation:
        # Transformations augmentées
        augmented_transforms = A.Compose(
            [
                A.Resize(height=args.train_h, width=args.train_w),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.2, rotate_limit=10, p=0.5
                ),
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(p=0.5),
                        A.RandomGamma(p=0.5),
                        A.CLAHE(p=0.5),
                    ],
                    p=0.5,
                ),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ],
            additional_targets={"mask": "mask"},
        )

        augmented_dataset = dataset(
            args.data_root,
            mode="train",
            transforms=augmented_transforms,
            transfo_activated=True,
        )

        # Concaténer les deux datasets
        train_data = torch.utils.data.ConcatDataset([train_data, augmented_dataset])

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
    )

    if args.class_weights:
        if args.weight_function == "enet":
            class_weights = enet_weighing(train_loader, args.classes)
        elif args.weight_function == "median":
            class_weights = median_freq_balancing(train_loader, args.classes)
            #  tensor([ 0.0817,  0.3042,  1.0811,  1.0000,  1.1928,  1.1088,  7.9330,  0.4916, 0.7801,  0.1408, 10.6358])

        else:
            class_weights = calculate_class_weights(train_loader, args.classes)

        if main_process():
            logger.info(f"Class weights: {class_weights}")

        # Initialize criterion with class weights
        criterion = nn.CrossEntropyLoss(
            weight=class_weights, ignore_index=args.ignore_label
        )
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    if args.arch == "pspnet":
        from models.pspnet_for_train import PSPNet

        model = PSPNet(
            layers=args.layers,
            classes=args.classes,
            zoom_factor=args.zoom_factor,
            criterion=criterion,
            BatchNorm=BatchNorm,
            pretrained=args.use_pretrained_weights,
        )
        modules_ori = [
            model.layer0,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        ]
        modules_new = [model.ppm, model.cls, model.aux]
    elif args.arch == "psa":
        from models.psanet import PSANet

        model = PSANet(
            layers=args.layers,
            classes=args.classes,
            zoom_factor=args.zoom_factor,
            psa_type=args.psa_type,
            compact=args.compact,
            shrink_factor=args.shrink_factor,
            mask_h=args.mask_h,
            mask_w=args.mask_w,
            normalization_factor=args.normalization_factor,
            psa_softmax=args.psa_softmax,
            criterion=criterion,
            BatchNorm=BatchNorm,
        )
        modules_ori = [
            model.layer0,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        ]
        modules_new = [model.psa, model.cls, model.aux]
    elif args.arch == "aunet":
        from models.unet import AttU_Net

        model = AttU_Net(img_ch=3, output_ch=args.classes)
    elif args.arch == "transformer":
        model = create_segmenter(args, criterion=criterion)
        params_list.append(dict(params=model.parameters(), lr=args.base_lr))
        args.index_split = 0

    if args.arch != "transformer":
        for module in modules_ori:
            params_list.append(dict(params=module.parameters(), lr=args.base_lr))
        for module in modules_new:
            params_list.append(dict(params=module.parameters(), lr=args.base_lr * 10))
        args.index_split = 5

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            params_list,
            lr=args.base_lr,  # 0.001 est pas mal
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            params_list,
            lr=args.base_lr,  # 0.00001 avec 0 de weight decay fonctionne "bien"
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),  # avec AdamW, pas de momentum -> betas
        )
    else:
        raise RuntimeError('"{0}" is not a supported optimizer.'.format(args.optimizer))

    args.save_path = (
        args.save_path + str(args.layers) + "/model"
    )  # essai pour avoir des dossier dynamiques
    if main_process():
        # logger = get_logger()
        writer = SummaryWriter(args.save_path)
        # logger.info(args)
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))
        # logger.info(model)

    model = torch.nn.DataParallel(
        model.cuda()
    )  # @sh: add to avoid prev commented out block

    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint["state_dict"])
            if main_process():
                logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            if main_process():
                logger.info("=> no weight found at '{}'".format(args.weight))

    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            # checkpoint = torch.load(args.resume)
            checkpoint = torch.load(
                args.resume, map_location=lambda storage, loc: storage.cuda()
            )
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            if main_process():
                logger.info(
                    "=> loaded checkpoint '{}' (epoch {})".format(
                        args.resume, checkpoint["epoch"]
                    )
                )
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    # Import the requested dataset
    if args.dataset.lower() != "rescuenet":
        # Should never happen...but just in case it does
        raise RuntimeError('"{0}" is not a supported dataset.'.format(args.dataset))

    if args.evaluate:
        label_transform = transforms.Compose(
            [
                transforms.Resize((args.train_h, args.train_w), Image.NEAREST),
                ext_transforms.PILToLongTensor(),
            ]
        )
        val_data = dataset(args.data_root, mode="val", transforms=base_transforms)
        val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=args.batch_size_val,
            shuffle=False,
            num_workers=args.workers,
        )
    print("Model device:", next(model.parameters()).device)

    for epoch in tqdm(range(args.start_epoch, args.epochs), desc="Epochs"):
        epoch_log = epoch + 1
        logger.info("Epoch " + str(epoch_log))

        loss_train, mIoU_train, mAcc_train, allAcc_train = train(
            train_loader, model, optimizer, epoch
        )
        # record train loss and miou corresponding to each epoch
        train_epochs.append(epoch)
        train_loss.append(loss_train)
        train_accuracy.append(mIoU_train)
        if main_process():
            writer.add_scalar("loss_train", loss_train, epoch_log)
            writer.add_scalar("mIoU_train", mIoU_train, epoch_log)
            writer.add_scalar("mAcc_train", mAcc_train, epoch_log)
            writer.add_scalar("allAcc_train", allAcc_train, epoch_log)

        if (epoch_log % args.save_freq == 0) and main_process():
            filename = args.save_path + "/train_epoch_" + str(epoch_log) + ".pth"
            logger.info("Saving checkpoint to: " + filename)
            torch.save(
                {
                    "epoch": epoch_log,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                filename,
            )
            if epoch_log / args.save_freq > 2:
                deletename = (
                    args.save_path
                    + "/train_epoch_"
                    + str(epoch_log - args.save_freq * 2)
                    + ".pth"
                )
                os.remove(deletename)
        if args.evaluate:
            print("Evaluation epoch ", epoch_log)
            loss_val, mIoU_val, mAcc_val, allAcc_val = validate(
                val_loader, model, criterion
            )
            # record validation loss and miou corresponding to each epoch
            val_epochs.append(epoch)
            val_loss.append(loss_val)
            val_accuracy.append(mIoU_val)
            if main_process():
                writer.add_scalar("loss_val", loss_val, epoch_log)
                writer.add_scalar("mIoU_val", mIoU_val, epoch_log)
                writer.add_scalar("mAcc_val", mAcc_val, epoch_log)
                writer.add_scalar("allAcc_val", allAcc_val, epoch_log)
    # Tracer la perte d'entraînement
    plt.figure()
    plt.plot(train_epochs, train_loss, label="Train Loss")
    plt.plot(val_epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss per Epoch")
    plt.show()

    # Tracer la précision d'entraînement (IoU ici)
    plt.figure()
    plt.plot(train_epochs, train_accuracy, label="Train Accuracy (mIoU)")
    plt.plot(val_epochs, val_accuracy, label="Validation Accuracy (mIoU)")
    plt.xlabel("Epochs")
    plt.ylabel("mIoU")
    plt.legend()
    plt.title("mIoU per Epoch")
    plt.show()


def train(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    main_loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    for i, (input, target) in tqdm(
        enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}"
    ):
        # for i, (input, target) in enumerate(train_loader):
        # print("dans la boucle for de train(), itération ",i)
        data_time.update(time.time() - end)
        if args.zoom_factor != 8:
            h = int((target.size()[1] - 1) / 8 * args.zoom_factor + 1)
            w = int((target.size()[2] - 1) / 8 * args.zoom_factor + 1)
            # 'nearest' mode doesn't support align_corners mode and 'bilinear' mode is fine for downsampling
            target = (
                F.interpolate(
                    target.unsqueeze(1).float(),
                    size=(h, w),
                    mode="bilinear",
                    align_corners=True,
                )
                .squeeze(1)
                .long()
            )

        input = input.cuda(non_blocking=True).float()
        target = target.cuda(non_blocking=True)
        if args.arch == "transformer":
            output = model(input)
            main_loss = F.cross_entropy(output, target, ignore_index=255)
            aux_loss = torch.tensor(
                0.0
            ).cuda()  # pas de perte auxiliaire pour le transformer
        else:  # pour pspnet et le reste
            output, main_loss, aux_loss = model(input, target)
            if not args.multiprocessing_distributed:
                main_loss, aux_loss = torch.mean(main_loss), torch.mean(aux_loss)

        loss = main_loss + args.aux_weight * aux_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n = input.size(0)
        if args.multiprocessing_distributed:
            main_loss, aux_loss, loss = (
                main_loss.detach() * n,
                aux_loss * n,
                loss * n,
            )  # not considering ignore pixels
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(main_loss), dist.all_reduce(aux_loss), dist.all_reduce(
                loss
            ), dist.all_reduce(count)
            n = count.item()
            main_loss, aux_loss, loss = main_loss / n, aux_loss / n, loss / n

        if args.arch == "transformer":
            output = output.max(1)[1]  # [B, H, W]

        intersection, union, target = intersectionAndUnionGPU(
            output, target, args.classes, args.ignore_label
        )
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
                target
            )
        intersection, union, target = (
            intersection.cpu().numpy(),
            union.cpu().numpy(),
            target.cpu().numpy(),
        )
        intersection_meter.update(intersection), union_meter.update(
            union
        ), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        main_loss_meter.update(main_loss.item(), n)
        aux_loss_meter.update(aux_loss.item(), n)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

        current_iter = epoch * len(train_loader) + i + 1
        current_lr = poly_learning_rate(
            args.base_lr, current_iter, max_iter, power=args.power
        )
        for index in range(0, args.index_split):
            optimizer.param_groups[index]["lr"] = current_lr
        for index in range(args.index_split, len(optimizer.param_groups)):
            optimizer.param_groups[index]["lr"] = current_lr * 10
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = "{:02d}:{:02d}:{:02d}".format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info(
                "Epoch: [{}/{}][{}/{}] "
                "Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Remain {remain_time} "
                "MainLoss {main_loss_meter.val:.4f} "
                "AuxLoss {aux_loss_meter.val:.4f} "
                "Loss {loss_meter.val:.4f} "
                "Accuracy {accuracy:.4f}.".format(
                    epoch + 1,
                    args.epochs,
                    i + 1,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    remain_time=remain_time,
                    main_loss_meter=main_loss_meter,
                    aux_loss_meter=aux_loss_meter,
                    loss_meter=loss_meter,
                    accuracy=accuracy,
                )
            )
        if main_process():
            writer.add_scalar("loss_train_batch", main_loss_meter.val, current_iter)
            writer.add_scalar(
                "mIoU_train_batch",
                np.mean(intersection / (union + 1e-10)),
                current_iter,
            )
            writer.add_scalar(
                "mAcc_train_batch",
                np.mean(intersection / (target + 1e-10)),
                current_iter,
            )
            writer.add_scalar("allAcc_train_batch", accuracy, current_iter)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info(
            "Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                epoch + 1, args.epochs, mIoU, mAcc, allAcc
            )
        )
    return main_loss_meter.avg, mIoU, mAcc, allAcc


def validate(val_loader, model, criterion):
    if main_process():
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        data_time.update(time.time() - end)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        output = model(input)
        if args.zoom_factor != 8:
            output = F.interpolate(
                output, size=target.size()[1:], mode="bilinear", align_corners=True
            )
        loss = criterion(output, target)

        n = input.size(0)
        if args.multiprocessing_distributed:
            loss = loss * n  # not considering ignore pixels
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss = loss / n
        else:
            loss = torch.mean(loss)

        output = output.max(1)[1]
        intersection, union, target = intersectionAndUnionGPU(
            output, target, args.classes, args.ignore_label
        )
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
                target
            )
        intersection, union, target = (
            intersection.cpu().numpy(),
            union.cpu().numpy(),
            target.cpu().numpy(),
        )
        intersection_meter.update(intersection), union_meter.update(
            union
        ), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if ((i + 1) % args.print_freq == 0) and main_process():
            logger.info(
                "Test: [{}/{}] "
                "Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) "
                "Accuracy {accuracy:.4f}.".format(
                    i + 1,
                    len(val_loader),
                    data_time=data_time,
                    batch_time=batch_time,
                    loss_meter=loss_meter,
                    accuracy=accuracy,
                )
            )

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                mIoU, mAcc, allAcc
            )
        )
        for i in range(args.classes):
            logger.info(
                "Class_{} Result: iou/accuracy {:.4f}/{:.4f}.".format(
                    i, iou_class[i], accuracy_class[i]
                )
            )
        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
    return loss_meter.avg, mIoU, mAcc, allAcc


"""
def test(model, test_loader, class_weights, class_encoding):
    print("\nTesting...\n")

    num_classes = len(class_encoding)

    # We are going to use the CrossEntropyLoss loss function as it's most
    # frequentely used in classification problems with multiple classes which
    # fits the problem. This criterion  combines LogSoftMax and NLLLoss.
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Evaluation metric
    if args.ignore_unlabeled:
        ignore_index = list(class_encoding).index("unlabeled")
    else:
        ignore_index = None
    metric = IoU(num_classes, ignore_index=ignore_index)

    # Test the trained model on the test set
    test = Test(model, test_loader, criterion, metric, device)

    print(">>>> Running test dataset")

    loss, (iou, miou) = test.run_epoch(args.print_step)
    class_iou = dict(zip(class_encoding.keys(), iou))

    print(">>>> Avg. loss: {0:.4f} | Mean IoU: {1:.4f}".format(loss, miou))

    # Print per class IoU
    for key, class_iou in zip(class_encoding.keys(), iou):
        print("{0}: {1:.4f}".format(key, class_iou))

    # Show a batch of samples and labels
    if args.imshow_batch:
        print("A batch of predictions from the test set...")
        images, _ = iter(test_loader).next()
        predict(model, images, class_encoding)


def predict(model, images, class_encoding):
    images = images.to(device)

    # Make predictions!
    model.eval()
    with torch.no_grad():
        predictions = model(images)

    # Predictions is one-hot encoded with "num_classes" channels.
    # Convert it to a single int using the indices where the maximum (1) occurs
    _, predictions = torch.max(predictions.data, 1)

    label_to_rgb = transforms.Compose(
        [ext_transforms.LongTensorToRGBPIL(class_encoding), transforms.ToTensor()]
    )
    color_predictions = utils.batch_transform(predictions.cpu(), label_to_rgb)
    utils.imshow_batch(images.data.cpu(), color_predictions)

"""
# Run only if this module is being run directly
if __name__ == "__main__":
    main()
