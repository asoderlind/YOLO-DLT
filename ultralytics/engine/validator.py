# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Check a model's accuracy on a test or val split of a dataset.

Usage:
    $ yolo mode=val model=yolo11n.pt data=coco8.yaml imgsz=640

Usage - formats:
    $ yolo mode=val model=yolo11n.pt                 # PyTorch
                          yolo11n.torchscript        # TorchScript
                          yolo11n.onnx               # ONNX Runtime or OpenCV DNN with dnn=True
                          yolo11n_openvino_model     # OpenVINO
                          yolo11n.engine             # TensorRT
                          yolo11n.mlpackage          # CoreML (macOS-only)
                          yolo11n_saved_model        # TensorFlow SavedModel
                          yolo11n.pb                 # TensorFlow GraphDef
                          yolo11n.tflite             # TensorFlow Lite
                          yolo11n_edgetpu.tflite     # TensorFlow Edge TPU
                          yolo11n_paddle_model       # PaddlePaddle
                          yolo11n.mnn                # MNN
                          yolo11n_ncnn_model         # NCNN
                          yolo11n_imx_model          # Sony IMX
                          yolo11n_rknn_model         # Rockchip RKNN
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import LOGGER, TQDM, callbacks, colorstr, emojis
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import de_parallel, select_device, smart_inference_mode


class BaseValidator:
    """
    BaseValidator.

    A base class for creating validators.

    Attributes:
        args (SimpleNamespace): Configuration for the validator.
        dataloader (DataLoader): Dataloader to use for validation.
        pbar (tqdm): Progress bar to update during validation.
        model (nn.Module): Model to validate.
        data (dict): Data dictionary.
        device (torch.device): Device to use for validation.
        batch_i (int): Current batch index.
        training (bool): Whether the model is in training mode.
        names (dict): Class names.
        seen: Records the number of images seen so far during validation.
        stats: Placeholder for statistics during validation.
        confusion_matrix: Placeholder for a confusion matrix.
        nc: Number of classes.
        iouv: (torch.Tensor): IoU thresholds from 0.50 to 0.95 in spaces of 0.05.
        jdict (dict): Dictionary to store JSON validation results.
        speed (dict): Dictionary with keys 'preprocess', 'inference', 'loss', 'postprocess' and their respective
                      batch processing times in milliseconds.
        save_dir (Path): Directory to save results.
        plots (dict): Dictionary to store plots for visualization.
        callbacks (dict): Dictionary to store various callback functions.
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """
        Initializes a BaseValidator instance.

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader to be used for validation.
            save_dir (Path, optional): Directory to save results.
            pbar (tqdm.tqdm): Progress bar for displaying progress.
            args (SimpleNamespace): Configuration for the validator.
            _callbacks (dict): Dictionary to store various callback functions.
        """
        self.args = get_cfg(overrides=args)
        self.dataloader = dataloader
        self.pbar = pbar
        self.stride = None
        self.data = None
        self.device = None
        self.batch_i = None
        self.training = True
        self.names = None
        self.seen = None
        self.stats = None
        self.confusion_matrix = None
        self.nc = None
        self.iouv = None
        self.jdict = None
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}

        self.save_dir = save_dir or get_save_dir(self.args)
        (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
        if self.args.conf is None:
            self.args.conf = 0.001  # default conf=0.001
        self.args.imgsz = check_imgsz(self.args.imgsz, max_dim=1)

        self.plots = {}
        self.callbacks = _callbacks or callbacks.get_default_callbacks()

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """Executes validation process, running inference on dataloader and computing performance metrics."""
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)
        # Initialize NMS analysis tracking
        self.nms_analysis = {
            "before_nms": [],
            "after_nms": [],
            "filtered_by_conf": [],
            "filtered_by_nms": [],
            "conf_distribution": [],
            "detection_counts": [],
        }
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            # force FP16 val during training
            self.args.half = self.device.type != "cpu" and trainer.amp
            model = trainer.ema.ema or trainer.model
            model = model.half() if self.args.half else model.float()
            # self.model = model
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()
        else:
            self.nms_log_file = f"nms_analysis_{self.args.name}.tsv"
            with open(self.nms_log_file, "w") as f:
                headers = [
                    "batch_idx",
                    "image_idx",
                    "stage",
                    "num_detections",
                    "num_enhanced",
                    "num_raw",
                    "conf_thresh",
                    "avg_conf_before",
                    "avg_conf_after",
                    "min_conf_before",
                    "max_conf_before",
                    "min_conf_after",
                    "max_conf_after",
                    "filtered_by_conf",
                    "filtered_by_nms",
                    "precision_change",
                    "recall_change",
                ]
                f.write("\t".join(headers) + "\n")
            if str(self.args.model).endswith(".yaml") and model is None:
                LOGGER.warning("WARNING ⚠️ validating an untrained model YAML will result in 0 mAP.")
            callbacks.add_integration_callbacks(self)
            model = AutoBackend(
                weights=model or self.args.model,
                device=select_device(self.args.device, self.args.batch),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half,
            )
            # self.model = model
            self.device = model.device  # update device
            self.args.half = model.fp16  # update half
            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if engine:
                self.args.batch = model.batch_size
            elif not pt and not jit:
                self.args.batch = model.metadata.get("batch", 1)  # export.py models default to batch-size 1
                LOGGER.info(f"Setting batch={self.args.batch} input of shape ({self.args.batch}, 3, {imgsz}, {imgsz})")

            if str(self.args.data).split(".")[-1] in {"yaml", "yml"}:
                self.data = check_det_dataset(self.args.data)
            elif self.args.task == "classify":
                self.data = check_cls_dataset(self.args.data, split=self.args.split)
            else:
                raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

            if self.device.type in {"cpu", "mps"}:
                self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
            if not pt:
                self.args.rect = False
            self.stride = model.stride  # used in get_dataloader() for padding
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)

            model.eval()
            model.warmup(imgsz=(1 if pt else self.args.batch, 3, imgsz, imgsz))  # warmup

        self.run_callbacks("on_val_start")
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(de_parallel(model))
        self.jdict = []  # empty before each val
        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            # Preprocess
            with dt[0]:
                batch = self.preprocess(batch)

            # Inference
            with dt[1]:
                preds = model(batch["img"], augment=augment)

            # Loss
            with dt[2]:
                if self.training:
                    self.loss += model.loss(batch, preds, is_validation=True)[1]

            # Enhanced NMS Analysis
            with dt[3]:
                if isinstance(preds, list) and len(preds) >= 4:
                    # Extract predictions
                    enhanced_preds = preds[0]  # Enhanced predictions
                    raw_preds = preds[3]  # Raw predictions

                    # Analyze before NMS
                    self._analyze_pre_nms(enhanced_preds, raw_preds, batch_i)

                    # Apply NMS to both
                    enhanced_output = self.postprocess((enhanced_preds,))
                    raw_output = self.postprocess((raw_preds,))

                    # Analyze after NMS
                    self._analyze_post_nms(enhanced_output, raw_output, enhanced_preds, raw_preds, batch_i)

                    # Use enhanced predictions for metrics
                    preds = enhanced_output
                else:
                    # Standard postprocess for non-temporal models
                    preds = self.postprocess(preds)

            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)

            self.run_callbacks("on_val_batch_end")
        stats = self.get_stats()
        self.check_stats(stats)
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks("on_val_end")
        if self.training:
            model.float()
            results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}
            return {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
        else:
            LOGGER.info(
                "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
                    *tuple(self.speed.values())
                )
            )
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / "predictions.json"), "w") as f:
                    LOGGER.info(f"Saving {f.name}...")
                    json.dump(self.jdict, f)  # flatten and save
                stats = self.eval_json(stats)  # update stats
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")

            # Save NMS analysis summary
            if not self.training:
                self._save_nms_analysis_summary()
            return stats

    def _analyze_pre_nms(self, enhanced_preds, raw_preds, batch_idx):
        """Analyze predictions before NMS is applied."""
        bs = enhanced_preds.shape[0]

        for img_idx in range(bs):
            # Get predictions for this image
            enh_pred = enhanced_preds[img_idx]  # [nc+4, num_anchors]
            raw_pred = raw_preds[img_idx]

            # Extract confidence scores (max class probability)
            enh_conf = enh_pred[4 : 4 + self.nc].max(0)[0]  # [num_anchors]
            raw_conf = raw_pred[4 : 4 + self.nc].max(0)[0]

            # Count predictions above confidence threshold
            enh_above_thresh = (enh_conf > self.args.conf).sum().item()
            raw_above_thresh = (raw_conf > self.args.conf).sum().item()

            # Store statistics
            self.nms_analysis["before_nms"].append(
                {
                    "batch_idx": batch_idx,
                    "img_idx": img_idx,
                    "enhanced_total": len(enh_conf),
                    "enhanced_above_conf": enh_above_thresh,
                    "raw_above_conf": raw_above_thresh,
                    "enhanced_conf_mean": enh_conf.mean().item(),
                    "raw_conf_mean": raw_conf.mean().item(),
                    "enhanced_conf_std": enh_conf.std().item(),
                    "raw_conf_std": raw_conf.std().item(),
                }
            )

    def _analyze_post_nms(self, enhanced_output, raw_output, enhanced_preds, raw_preds, batch_idx):
        """Analyze predictions after NMS is applied."""
        bs = len(enhanced_output)

        for img_idx in range(bs):
            enh_dets = enhanced_output[img_idx]  # [n_dets, 6+]
            raw_dets = raw_output[img_idx]

            # Count detections
            n_enh = len(enh_dets)
            n_raw = len(raw_dets)

            # Get confidence statistics after NMS
            enh_confs = enh_dets[:, 4] if n_enh > 0 else torch.tensor([])
            raw_confs = raw_dets[:, 4] if n_raw > 0 else torch.tensor([])

            # Analyze filtering
            pre_nms_stats = self.nms_analysis["before_nms"][-1]
            filtered_by_conf_enh = pre_nms_stats["enhanced_total"] - pre_nms_stats["enhanced_above_conf"]
            filtered_by_nms_enh = pre_nms_stats["enhanced_above_conf"] - n_enh

            # Log to TSV
            if not self.training:
                with open(self.nms_log_file, "a") as f:
                    row = [
                        batch_idx,
                        img_idx,
                        "post_nms",
                        n_enh,
                        n_enh,
                        n_raw,
                        self.args.conf,
                        pre_nms_stats["enhanced_conf_mean"],
                        enh_confs.mean().item() if n_enh > 0 else 0,
                        pre_nms_stats["enhanced_conf_mean"] - 2 * pre_nms_stats["enhanced_conf_std"],
                        pre_nms_stats["enhanced_conf_mean"] + 2 * pre_nms_stats["enhanced_conf_std"],
                        enh_confs.min().item() if n_enh > 0 else 0,
                        enh_confs.max().item() if n_enh > 0 else 0,
                        filtered_by_conf_enh,
                        filtered_by_nms_enh,
                        (n_enh - n_raw) / max(n_raw, 1),  # relative change
                        0,  # placeholder for recall change
                    ]
                    f.write("\t".join(map(str, row)) + "\n")

            self.nms_analysis["after_nms"].append(
                {
                    "batch_idx": batch_idx,
                    "img_idx": img_idx,
                    "enhanced_count": n_enh,
                    "raw_count": n_raw,
                    "count_diff": n_enh - n_raw,
                    "enhanced_conf_mean": enh_confs.mean().item() if n_enh > 0 else 0,
                    "raw_conf_mean": raw_confs.mean().item() if n_raw > 0 else 0,
                    "filtered_by_conf": filtered_by_conf_enh,
                    "filtered_by_nms": filtered_by_nms_enh,
                }
            )

    def _save_nms_analysis_summary(self):
        """Save summary statistics of NMS analysis."""
        import pandas as pd

        # Convert to DataFrame for easier analysis
        df_after = pd.DataFrame(self.nms_analysis["after_nms"])
        summary = {
            "total_images": len(df_after),
            "avg_enhanced_detections": df_after["enhanced_count"].mean(),
            "avg_raw_detections": df_after["raw_count"].mean(),
            "avg_detection_diff": df_after["count_diff"].mean(),
            "images_with_fewer_enhanced": (df_after["count_diff"] < 0).sum(),
            "images_with_more_enhanced": (df_after["count_diff"] > 0).sum(),
            "avg_filtered_by_conf": df_after["filtered_by_conf"].mean(),
            "avg_filtered_by_nms": df_after["filtered_by_nms"].mean(),
        }

        # Save summary
        with open(f"nms_analysis_summary_{self.args.name}.txt", "w") as f:
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")

        LOGGER.info("NMS Analysis Summary saved. Key findings:")
        LOGGER.info(
            f"  - Avg detections: Enhanced={summary['avg_enhanced_detections']:.1f}, Raw={summary['avg_raw_detections']:.1f}"
        )
        LOGGER.info(f"  - Images with fewer enhanced detections: {summary['images_with_fewer_enhanced']}")
        LOGGER.info(f"  - Avg filtered by confidence: {summary['avg_filtered_by_conf']:.1f}")
        LOGGER.info(f"  - Avg filtered by NMS: {summary['avg_filtered_by_nms']:.1f}")

    def match_predictions(self, pred_classes, true_classes, iou, use_scipy=False):
        """
        Matches predictions to ground truth objects (pred_classes, true_classes) using IoU.

        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape(N,).
            true_classes (torch.Tensor): Target class indices of shape(M,).
            iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground of truth
            use_scipy (bool): Whether to use scipy for matching (more precise).

        Returns:
            (torch.Tensor): Correct tensor of shape(N,10) for 10 IoU thresholds.
        """
        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        # LxD matrix where L - labels (rows), D - detections (columns)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # zero out the wrong classes
        iou = iou.cpu().numpy()
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            if use_scipy:
                # WARNING: known issue that reduces mAP in https://github.com/ultralytics/ultralytics/pull/4708
                import scipy  # scope import to avoid importing for all commands

                cost_matrix = iou * (iou >= threshold)
                if cost_matrix.any():
                    labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix)
                    valid = cost_matrix[labels_idx, detections_idx] > 0
                    if valid.any():
                        correct[detections_idx[valid], i] = True
            else:
                matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
                matches = np.array(matches).T
                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        # matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)

    def add_callback(self, event: str, callback):
        """Appends the given callback."""
        self.callbacks[event].append(callback)

    def run_callbacks(self, event: str):
        """Runs all callbacks associated with a specified event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def get_dataloader(self, dataset_path, batch_size):
        """Get data loader from dataset path and batch size."""
        raise NotImplementedError("get_dataloader function not implemented for this validator")

    def build_dataset(self, img_path):
        """Build dataset."""
        raise NotImplementedError("build_dataset function not implemented in validator")

    def preprocess(self, batch):
        """Preprocesses an input batch."""
        return batch

    def postprocess(self, preds):
        """Preprocesses the predictions."""
        return preds

    def init_metrics(self, model):
        """Initialize performance metrics for the YOLO model."""
        pass

    def update_metrics(self, preds, batch):
        """Updates metrics based on predictions and batch."""
        pass

    def finalize_metrics(self, *args, **kwargs):
        """Finalizes and returns all metrics."""
        pass

    def get_stats(self):
        """Returns statistics about the model's performance."""
        return {}

    def check_stats(self, stats):
        """Checks statistics."""
        pass

    def print_results(self):
        """Prints the results of the model's predictions."""
        pass

    def get_desc(self):
        """Get description of the YOLO model."""
        pass

    @property
    def metric_keys(self):
        """Returns the metric keys used in YOLO training/validation."""
        return []

    def on_plot(self, name, data=None):
        """Registers plots (e.g. to be consumed in callbacks)."""
        self.plots[Path(name)] = {"data": data, "timestamp": time.time()}

    # TODO: may need to put these following functions into callback
    def plot_val_samples(self, batch, ni):
        """Plots validation samples during training."""
        pass

    def plot_predictions(self, batch, preds, ni):
        """Plots YOLO model predictions on batch images."""
        pass

    def pred_to_json(self, preds, batch):
        """Convert predictions to JSON format."""
        pass

    def eval_json(self, stats):
        """Evaluate and return JSON format of prediction statistics."""
        pass
