# test.py
import os
import time
import logging
import argparse

import re
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data

from util import dataset, transform, config
from util.util import (
    AverageMeter,
    poly_learning_rate,
    intersectionAndUnionGPU,
    check_makedirs,
    colorize,
    enet_weighing,
)  # @sh: add
from evaluate import Test
from metric.iou import IoU
from skimage import io

cv2.ocl.setUseOpenCL(False)

from PIL import Image
import torchvision.transforms as transforms
import transforms as ext_transforms
import torch.nn as nn
import utils
from torchvision.utils import save_image

## to implement Transformer
from models.factory import create_segmenter

# pour tester
from models.pspnet_for_train import (
    PSPNet,
)  # Utilisez le même fichier que pour l'entraînement
from models import resnet_for_train


device = torch.device("cuda")


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


def check(args):
    assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]
    assert args.split in ["train", "val", "test"]
    if args.arch == "pspnet":
        assert (args.train_h - 1) % 8 == 0 and (args.train_w - 1) % 8 == 0
    elif args.arch == "psa":
        if args.compact:
            args.mask_h = (args.train_h - 1) // (8 * args.shrink_factor) + 1
            args.mask_w = (args.train_w - 1) // (8 * args.shrink_factor) + 1
        else:
            assert (args.mask_h is None and args.mask_w is None) or (
                args.mask_h is not None and args.mask_w is not None
            )
            if args.mask_h is None and args.mask_w is None:
                args.mask_h = (
                    2 * ((args.train_h - 1) // (8 * args.shrink_factor) + 1) - 1
                )
                args.mask_w = (
                    2 * ((args.train_w - 1) // (8 * args.shrink_factor) + 1) - 1
                )
            else:
                assert (
                    (args.mask_h % 2 == 1)
                    and (args.mask_h >= 3)
                    and (
                        args.mask_h
                        <= 2 * ((args.train_h - 1) // (8 * args.shrink_factor) + 1) - 1
                    )
                )
                assert (
                    (args.mask_w % 2 == 1)
                    and (args.mask_w >= 3)
                    and (
                        args.mask_w
                        <= 2 * ((args.train_h - 1) // (8 * args.shrink_factor) + 1) - 1
                    )
                )
    elif args.arch == "transformer":
        print(f"{args.arch} utilisé")
    else:
        raise Exception("architecture not supported yet".format(args.arch))


def main():
    global args, logger
    args = get_parser()

    check(args)
    logger = get_logger()
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in args.test_gpu)
    logger.info(args)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))

    # device = torch.device(args.device)

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    gray_folder = os.path.join(args.save_folder, "gray")
    color_folder = os.path.join(args.save_folder, "color")

    # Import the requested dataset
    if args.dataset.lower() == "rescuenet":
        from data.rescuenet import RescueNet as dataset
    else:
        # Should never happen...but just in case it does
        raise RuntimeError('"{0}" is not a supported dataset.'.format(args.dataset))

    image_transform = transforms.Compose(
        [transforms.Resize((args.train_h, args.train_w)), transforms.ToTensor()]
    )

    label_transform = transforms.Compose(
        [
            transforms.Resize((args.train_h, args.train_w), Image.NEAREST),
            ext_transforms.PILToLongTensor(),
        ]
    )

    train_set = dataset(
        args.data_root, transform=image_transform, label_transform=label_transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
    )

    # Get encoding between pixel valus in label images and RGB colors
    class_encoding = train_set.color_encoding

    # Get number of classes to predict
    num_classes = len(class_encoding)

    # Print information for debugging
    print("Number of classes to predict:", num_classes)
    print("Train dataset size:", len(train_set))
    print("Class-color encoding:", class_encoding)

    # Get class weights from the selected weighing technique
    print("Computing class weights...")
    print("(this can take a while depending on the dataset size)")
    class_weights = 0
    class_weights = enet_weighing(train_loader, num_classes)

    if class_weights is not None:
        class_weights = torch.from_numpy(class_weights).float().to(device)
    print("Class weights:", class_weights)

    # Load the test set as tensors for visulation
    test_set_vis = dataset(
        args.data_root,
        mode="vis",
        transform=image_transform,
        label_transform=label_transform,
    )

    test_loader_vis = torch.utils.data.DataLoader(
        test_set_vis, batch_size=1, shuffle=False, num_workers=args.workers
    )

    test_data = dataset(
        args.data_root, transform=image_transform, label_transform=label_transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # colors = np.loadtxt(color_folder).astype('uint8')
    # names = [line.rstrip('\n') for line in open(args.names_path)]

    if not args.has_prediction:
        if args.arch == "pspnet":

            # Forcer deep_base=False car les poids sauvegardés utilisent l'architecture standard
            model = PSPNet(
                layers=args.layers,
                classes=args.classes,
                zoom_factor=args.zoom_factor,
                pretrained=args.use_pretrained_weights,
                # deep_base=False,
            )
        elif args.arch == "psa":
            from model.psanet import PSANet

            model = PSANet(
                layers=args.layers,
                classes=args.classes,
                zoom_factor=args.zoom_factor,
                compact=args.compact,
                shrink_factor=args.shrink_factor,
                mask_h=args.mask_h,
                mask_w=args.mask_w,
                normalization_factor=args.normalization_factor,
                psa_softmax=args.psa_softmax,
                pretrained=False,
            )
        elif args.arch == "aunet":
            from models.unet import AttU_Net

            model = AttU_Net(img_ch=3, output_ch=args.classes)
        elif args.arch == "transformer":
            model = create_segmenter(args)

        logger.info(model)
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True

        if os.path.isfile(args.model_path):
            logger.info("=> loading checkpoint '{}'".format(args.model_path))
            checkpoint = torch.load(args.model_path)
            logger.info("Saved weights structure:")
            """
            for key in checkpoint["state_dict"].keys():
                logger.info(f"{key}: {checkpoint['state_dict'][key].shape}")

            # Debug: afficher la structure du modèle actuel
            logger.info("Current model structure:")
            for name, param in model.named_parameters():
                logger.info(f"{name}: {param.shape}")
            """
            model.load_state_dict(checkpoint["state_dict"], strict=False)
            logger.info("=> loaded checkpoint '{}'".format(args.model_path))
        else:
            raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))
        if args.mode.lower() == "test":
            print(model)
            test(model, test_loader, class_weights, class_encoding)
        elif args.mode.lower() == "vis":
            print(model)
            test(model, test_loader_vis, class_weights, class_encoding)


def net_process(model, image, mean, std=None, flip=True):
    input = torch.from_numpy(image.transpose((2, 0, 1))).float()
    if std is None:
        for t, m in zip(input, mean):
            t.sub_(m)
    else:
        for t, m, s in zip(input, mean, std):
            t.sub_(m).div_(s)
    input = input.unsqueeze(0).cuda()
    if flip:
        input = torch.cat([input, input.flip(3)], 0)
    with torch.no_grad():
        output = model(input)
    _, _, h_i, w_i = input.shape
    _, _, h_o, w_o = output.shape
    if (h_o != h_i) or (w_o != w_i):
        output = F.interpolate(output, (h_i, w_i), mode="bilinear", align_corners=True)
    output = F.softmax(output, dim=1)
    if flip:
        output = (output[0] + output[1].flip(2)) / 2
    else:
        output = output[0]
    output = output.data.cpu().numpy()
    output = output.transpose(1, 2, 0)
    return output


def scale_process(
    model, image, classes, crop_h, crop_w, h, w, mean, std=None, stride_rate=2 / 3
):
    ori_h, ori_w, _ = image.shape
    pad_h = max(crop_h - ori_h, 0)
    pad_w = max(crop_w - ori_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(
            image,
            pad_h_half,
            pad_h - pad_h_half,
            pad_w_half,
            pad_w - pad_w_half,
            cv2.BORDER_CONSTANT,
            value=mean,
        )
    new_h, new_w, _ = image.shape
    stride_h = int(np.ceil(crop_h * stride_rate))
    stride_w = int(np.ceil(crop_w * stride_rate))
    grid_h = int(np.ceil(float(new_h - crop_h) / stride_h) + 1)
    grid_w = int(np.ceil(float(new_w - crop_w) / stride_w) + 1)
    prediction_crop = np.zeros((new_h, new_w, classes), dtype=float)
    count_crop = np.zeros((new_h, new_w), dtype=float)
    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            s_h = index_h * stride_h
            e_h = min(s_h + crop_h, new_h)
            s_h = e_h - crop_h
            s_w = index_w * stride_w
            e_w = min(s_w + crop_w, new_w)
            s_w = e_w - crop_w
            image_crop = image[s_h:e_h, s_w:e_w].copy()
            count_crop[s_h:e_h, s_w:e_w] += 1
            prediction_crop[s_h:e_h, s_w:e_w, :] += net_process(
                model, image_crop, mean, std
            )
    prediction_crop /= np.expand_dims(count_crop, 2)
    prediction_crop = prediction_crop[
        pad_h_half : pad_h_half + ori_h, pad_w_half : pad_w_half + ori_w
    ]
    prediction = cv2.resize(prediction_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    return prediction


def test(model, test_loader, class_weights, class_encoding):
    print("\nTesting...\n")

    num_classes = len(class_encoding)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    if args.ignore_unlabeled:
        ignore_index = list(class_encoding).index("unlabeled")
    else:
        ignore_index = None
    metric = IoU(num_classes, ignore_index=ignore_index)

    if args.mode.lower() == "test":
        # Create results directory if it doesn't exist
        results_dir = os.path.join(args.save_folder, "numeric_results")
        os.makedirs(results_dir, exist_ok=True)

        # Test the trained model
        test = Test(model, test_loader, criterion, metric, device)
        print(">>>> Running test dataset")
        loss, (iou, miou) = test.run_epoch(args.print_step)
        class_iou = dict(zip(class_encoding.keys(), iou))

        print(">>>> Avg. loss: {0:.4f} | Mean IoU: {1:.4f}".format(loss, miou))

        # Prepare results for saving
        results = {"loss": float(loss), "mean_iou": float(miou), "class_iou": {}}

        # Print and save per class IoU
        for key, class_iou in zip(class_encoding.keys(), iou):
            print("{0}: {1:.4f}".format(key, class_iou))
            results["class_iou"][key] = float(class_iou)

        # Save numeric results to files
        timestamp = time.strftime("%Hh%M%S_%d%m%Y")

        # Save detailed results in JSON format
        import json

        json_path = os.path.join(results_dir, f"results_{timestamp}.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4)

        # Save summary in text format
        txt_path = os.path.join(results_dir, f"summary_{timestamp}.txt")
        with open(txt_path, "w") as f:
            f.write(f"Test Results Summary\n")
            f.write(f"===================\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {args.model_path}\n\n")
            f.write(f"Using pretrained weights:{args.use_pretrained_weights}\n")
            f.write(f"Trained on {args.epochs} epochs\n")
            f.write(f"Using {args.arch} architecture\n\n")
            f.write(f"Average Loss: {loss:.4f}\n")
            f.write(f"Mean IoU: {miou:.4f}\n\n")
            f.write("Per-class IoU:\n")
            for key, value in results["class_iou"].items():
                f.write(f"{key}: {value:.4f}\n")

        print(f"\nNumeric results saved to:")
        print(f"- Detailed results: {json_path}")
        print(f"- Summary: {txt_path}")

    # Visual results processing
    if args.imshow_batch:
        print("Processing and saving visual predictions...")
        for _, batch_data in enumerate(test_loader):
            images = batch_data[0]
            paths = batch_data[1]
            predict(model, images, paths, class_encoding)


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
        ignore_index = list(class_encoding).index('unlabeled')
    else:
        ignore_index = None
    metric = IoU(num_classes, ignore_index=ignore_index)

    if args.mode.lower() == 'test':
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
        for _, batch_data in enumerate(test_loader):
            # Get the inputs and labels
            images = batch_data[0]
            paths = batch_data[1]           
            predict(model, images, paths, class_encoding)
"""


def predict_alt_claude(model, images, paths, class_encoding):
    """
    Make predictions on a batch of images and save the results

    Args:
        model: The trained model
        images: Batch of images to predict on
        paths: Paths to save the prediction results
        class_encoding: Dictionary mapping class labels to RGB values
    """
    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    # Ensure model is in eval mode
    model.eval()

    # Move images to GPU if available
    images = images.cuda()

    with torch.no_grad():
        # Forward pass
        if args.arch == "transformer":
            # Direct prediction for transformer models
            predictions = model(images)
        else:
            # Original prediction path for other architectures
            predictions = model(images)

            # Handle auxiliary outputs if present
            if isinstance(predictions, tuple):
                predictions = predictions[0]

        # Get class predictions
        _, predictions = torch.max(predictions.data, 1)

        # Setup color transformation if needed
        if args.predict_color:
            label_to_rgb = transforms.Compose(
                [
                    ext_transforms.LongTensorToRGBPIL(class_encoding),
                    transforms.ToTensor(),
                ]
            )

            # Transform predictions to RGB
            predictions = utils.batch_transform(predictions.cpu(), label_to_rgb)

        # Save predictions
        for predict, impath in zip(predictions, paths):
            # Create output filename
            outname = os.path.splitext(impath)[0] + ".png"
            outname_final = os.path.join(args.output, outname)

            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(outname_final), exist_ok=True)

            if args.predict_color:
                save_image(predict, outname_final)
            else:
                # For grayscale predictions, ensure proper format
                predict_np = predict.cpu().numpy()
                if len(predict_np.shape) > 2:
                    predict_np = predict_np.squeeze()
                io.imsave(outname_final, predict_np)


def predict(model, images, paths, class_encoding):
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    images = images.to(device)
    # Make predictions!
    model.eval()
    with torch.no_grad():
        # it = iter(images).next()
        predictions = model(images)

    # Predictions is one-hot encoded with "num_classes" channels.
    # Convert it to a single int using the indices where the maximum (1) occurs
    _, predictions = torch.max(predictions.data, 1)
    label_to_rgb = transforms.Compose(
        [ext_transforms.LongTensorToRGBPIL(class_encoding), transforms.ToTensor()]
    )
    # print(predictions.shape)

    if args.predict_color:
        predictions = utils.batch_transform(predictions.cpu(), label_to_rgb)
        # print(predictions.shape)
    # Visualize and dump the results
    im_paths = paths
    for predict, impath in zip(predictions, im_paths):
        outname = os.path.splitext(impath)[0] + ".png"
        outname_final = os.path.join(args.output, outname)
        if args.predict_color:
            save_image(predict, outname_final)
        else:
            io.imsave(outname_final, predict.cpu())


if __name__ == "__main__":
    main()
