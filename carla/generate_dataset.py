#!/usr/bin/env python

import argparse
import queue
import os
import glob
import time
import random
import numpy as np
import carla

cls2id = {
    'car': 0,
    'truck': 1,
    'van': 2,
    'bus': 3,
    'motorcycle': 4,
    'bicycle': 5,
}

MAX_DIST = 150

def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))

def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


def get_image_point(loc, K, w2c):
    # Calculate 2D projection of 3D coordinate

    # Format the input coordinate (loc is a carla.Position object)
    point = np.array([loc.x, loc.y, loc.z, 1])
    # transform to camera coordinates
    point_camera = np.dot(w2c, point)

    # New we must change from UE4's coordinate system to an "standard"
    # (x, y ,z) -> (y, -z, x)
    # and we remove the fourth componebonent also
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    # now project 3D->2D using the camera matrix
    point_img = np.dot(K, point_camera)
    # normalize
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img[0:2]


def main(client, frame_count, selected_map, train_set):
    print(f"Running with:\n- Map: {selected_map}\n- Frames: {frame_count}\n- Train set: {train_set}")
    # get unique id for each video
    video_id = str(int(time.time()))
    current_frame = 0
    world = client.load_world(selected_map)

    weather = world.get_weather()
    weather.sun_altitude_angle = -90
    world.set_weather(weather)

    bp_lib = world.get_blueprint_library()

    # Get the map spawn points
    spawn_points = world.get_map().get_spawn_points()

    # spawn vehicle
    vehicle_bp = bp_lib.find("vehicle.lincoln.mkz_2020")

    vehicle = None
    while not vehicle:
        vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
    vehicle.set_autopilot(True)

    # spawn camera
    camera_bp = bp_lib.find("sensor.camera.rgb")
    camera_init_trans = carla.Transform(carla.Location(z=2))
    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

    # Spawn other vehicles
    if len(world.get_actors().filter("*vehicle*")) < 50:
        print("Spawning vehicles")
        for i in range(50):
            vehicle_bp = random.choice(bp_lib.filter("vehicle"))
            npc = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
            if npc:
                npc.set_autopilot(True)

    # Set up the simulator in synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True  # Enables synchronous mode
    settings.fixed_delta_seconds = 0.05  # 20 FPS
    world.apply_settings(settings)

    # Create a queue to store and retrieve the sensor data
    image_queue = queue.Queue()
    camera.listen(image_queue.put)

    # Get the world to camera matrix
    world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

    # Get the attributes from the camera
    image_w = camera_bp.get_attribute("image_size_x").as_int()
    image_h = camera_bp.get_attribute("image_size_y").as_int()
    fov = camera_bp.get_attribute("fov").as_float()

    # Calculate the camera projection matrix to project from 3D -> 2D
    K = build_projection_matrix(image_w, image_h, fov)

    while True:
        # Retrieve the image
        world.tick()
        image = image_queue.get()
        output_folder = "/home/phoawb/repos/yolo-testing/datasets/carla-yolo"
        images_folder = os.path.join(output_folder, "images", train_set)
        labels_folder = os.path.join(output_folder, "labels", train_set)

        if not os.path.exists(images_folder):
            os.makedirs(images_folder)

        if not os.path.exists(labels_folder):
            os.makedirs(labels_folder)

        # Get the camera matrix
        world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

        frame_name = "vid_" + video_id + "_frame_%06d" % current_frame
        current_frame += 1

        frame_path_image = f"{images_folder}/{frame_name}.png"
        # Save the image
        image.save_to_disk(frame_path_image)
        frame_path_label = f"{labels_folder}/{frame_name}.txt"

        # Initialize the exporter
        output_lines = []

        for npc in world.get_actors().filter("*vehicle*"):
            if npc.id != vehicle.id:
                bb = npc.bounding_box
                dist = npc.get_transform().location.distance(vehicle.get_transform().location)
                if dist < MAX_DIST:
                    forward_vec = vehicle.get_transform().get_forward_vector()
                    ray = npc.get_transform().location - vehicle.get_transform().location
                    if forward_vec.dot(ray) > 1:
                        # p1 = get_image_point(bb.location, K, world_2_camera)
                        verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                        x_max = -10000
                        x_min = 10000
                        y_max = -10000
                        y_min = 10000
                        for vert in verts:
                            p = get_image_point(vert, K, world_2_camera)
                            if p[0] > x_max:
                                x_max = p[0]
                            if p[0] < x_min:
                                x_min = p[0]
                            if p[1] > y_max:
                                y_max = p[1]
                            if p[1] < y_min:
                                y_min = p[1]

                        # Add the object to the frame and clamp the values to the image size
                        # x_min = max(0, x_min)
                        # x_max = min(image_w, x_max)
                        # y_min = max(0, y_min)
                        # y_max = min(image_h, y_max)
                        if x_min > 0 and x_max < image_w and y_min > 0 and y_max < image_h:
                            x_center = ((x_min + x_max) / 2) / image_w
                            y_center = ((y_min + y_max) / 2) / image_h
                            box_width = (x_max - x_min) / image_w
                            box_height = (y_max - y_min) / image_h
                            if x_center > 1 or y_center > 1 or box_width > 1 or box_height > 1:
                                raise ValueError(f"One of x_c {x_center} y_c {y_center} bw {box_width} bh {box_height} are not normalized")
                            vehicle_type = bp_lib.find(npc.type_id).get_attribute('base_type').as_str()
                            vehicle_id = None
                            if npc.type_id == 'vehicle.bmw.grandtourer' or npc.type_id == 'vehicle.mini.cooper_s':
                                vehicle_id = 0
                            else:
                                vehicle_id = cls2id.get(vehicle_type.lower())
                            if vehicle_id is None:
                                raise ValueError(f"{npc.type_id} has no base_type")
                            dist = clamp(dist, 0, MAX_DIST) / MAX_DIST
                            if dist < 0 or dist > 1.0:
                                raise ValueError(f"Distance is not normalized: {dist}")
                            output_lines.append(
                                f"{vehicle_id} {x_center:6f} {y_center:.6f} {box_width:.6f} {box_height:.6f} {dist:.6f}"
                            )

        # Save the bounding boxes in the scene
        with open(frame_path_label, "w") as f:
            for line in output_lines:
                f.write(line + "\n")
        print(
            f"Saved image {frame_path_image} and label {frame_path_label}, total images:",
            len(glob.glob(images_folder + "/*.png")),
            ", total labels:",
            len(glob.glob(labels_folder + "/*.txt")),
        )

        if current_frame >= frame_count:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CARLA simulation with configurable settings.")
    parser.add_argument("--frames", type=int, default=500, help="Number of frames to run")
    parser.add_argument("--map", type=str, help="Specify a CARLA map (random if not provided)")
    parser.add_argument("--train_set", type=str, default="train", help="Specify the training set")
    parser.add_argument("--iterations", type=int, default=1, help="re-run number of times")

    args = parser.parse_args()

    client = carla.Client("localhost", 2000)
    available_maps = client.get_available_maps()

    for map_name in available_maps:
        print("Available map:", map_name)

    for i in range(args.iterations):
        print(f"Iteration {i+1} out of {args.iterations}")
        selected_map = args.map if args.map else random.choice(available_maps)
        main(client, args.frames, selected_map, args.train_set)
