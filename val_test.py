from ultralytics import YOLO

model_bic_rfaconv = YOLO("bic-repc3k2.pt")


model_bic_rfaconv.val(
    data="bdd100k_night.yaml",
    device="cuda",
    batch=1,
    conf=0.25,
)
