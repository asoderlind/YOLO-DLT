def profile_model_with_dummy_conv(yaml_path, imgsz=640):
    """
    Profile a YOLO model by temporarily inserting a dummy 1x1 conv layer at the beginning.

    Args:
        yaml_path: Path to the YAML model file
        imgsz: Image size for profiling (default: 640)

    Returns:
        float: Model FLOPs
    """
    import os
    import tempfile

    import yaml

    from ultralytics.nn.tasks import yaml_model_load
    from ultralytics.utils.torch_utils import get_flops

    # Load the original YAML model
    model_dict = yaml_model_load(yaml_path)
    # Insert dummy 1x1 conv at the beginning of the backbone
    dummy_conv_schema = [-1, 1, "Conv", [3, 1, 1, None, 1, 1, False]]
    model_dict["backbone"].insert(0, dummy_conv_schema)

    # Update indices in the backbone and head

    # Find indices that are referenced in Concat and Detect operations
    for layer in model_dict["head"]:
        if isinstance(layer[0], list):  # References to previous layers (like in Concat or Detect)
            for i, idx in enumerate(layer[0]):
                if idx < 0:
                    # Negative indices (-1 etc.) don't need adjustment as they're relative
                    continue
                # Positive indices refer to absolute positions
                layer[0][i] = idx + 1  # Adjust for the inserted layer
        elif isinstance(layer[0], int) and layer[0] >= 0:  # Direct positive index references
            layer[0] += 1
    # Create a temporary file to save the modified YAML
    tmp_prefix = yaml_path.split("/")[-1].split(".")[0]
    fd, temp_yaml_path = tempfile.mkstemp(suffix=".yaml", prefix=tmp_prefix)
    os.close(fd)

    try:
        # Save the modified model to the temporary file
        with open(temp_yaml_path, "w") as f:
            # Convert the dict back to YAML format
            yaml.dump(model_dict, f, sort_keys=False)

        # Create and profile the model
        from ultralytics import YOLO

        model = YOLO(temp_yaml_path)
        model.info()
        flops = get_flops(model.model, imgsz)

        return flops

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_yaml_path):
            os.remove(temp_yaml_path)


profile_model_with_dummy_conv("dlt-models/yolo11n-spdconv-4.yaml")

# model = YOLO("dlt-models/yolo11n-spdconv-5.yaml")

# model.info()
