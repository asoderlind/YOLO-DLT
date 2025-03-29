#!/usr/bin/env python

import queue
import os
import glob
import time
import random
import numpy as np
import carla

FRAME_COUNT = 500


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


def main(client, selected_map="Town01"):
    # get unique id for each video
    video_id = str(int(time.time()))
    current_frame = 0
    world = client.load_world(selected_map)
    print("Selected map:", selected_map)

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
        images_folder = os.path.join(output_folder, "images")
        labels_folder = os.path.join(output_folder, "labels")

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
                if dist < 50:
                    forward_vec = vehicle.get_transform().get_forward_vector()
                    ray = npc.get_transform().location - vehicle.get_transform().location
                    if forward_vec.dot(ray) > 1:
                        p1 = get_image_point(bb.location, K, world_2_camera)
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
                            output_lines.append(
                                f"0 {x_center:6f} {y_center:.6f} {box_width:.6f} {box_height:.6f} {dist:.6f}"
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

        if current_frame > FRAME_COUNT:
            break


if __name__ == "__main__":
    client = carla.Client("localhost", 2000)
    available_maps = client.get_available_maps()
    for map_name in available_maps:
        print("Available map:", map_name)
    selected_map = random.choice(available_maps)
    main(client, selected_map)
