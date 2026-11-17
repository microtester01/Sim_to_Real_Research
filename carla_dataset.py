import argparse
import json
import math
import queue
import random
from collections import Counter
from pathlib import Path

import carla
import numpy as np
from PIL import Image


CLASS_IDS = {"Pedestrian": 0, "Cyclist": 1, "IgnoreDontCare": 2}
RGB_SIZE = (1280, 720)
WEATHER_NAMES = ("ClearNoon", "MidRainySunset", "HeavyRainNoon")


def bbox_iou(first, second):
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    area_first = (first[2] - first[0] + 1) * (first[3] - first[1] + 1)
    area_second = (second[2] - second[0] + 1) * (
        second[3] - second[1] + 1
    )
    union = area_first + area_second - intersection
    return intersection / float(union) if union else 0.0


def project_bbox(actor, camera, width, height, horizontal_fov):
    inverse = camera.get_transform().get_inverse_matrix()
    focal_length = width / (
        2.0 * math.tan(math.radians(horizontal_fov) / 2.0)
    )
    projected = []
    for vertex in actor.bounding_box.get_world_vertices(actor.get_transform()):
        point = (vertex.x, vertex.y, vertex.z, 1.0)
        camera_point = [
            sum(inverse[row][column] * point[column] for column in range(4))
            for row in range(3)
        ]
        if camera_point[0] <= 0.01:
            continue
        x = focal_length * camera_point[1] / camera_point[0] + width / 2.0
        y = height / 2.0 - focal_length * camera_point[2] / camera_point[0]
        projected.append((x, y))
    if not projected:
        return None
    box = [
        max(0, int(math.floor(min(point[0] for point in projected)))),
        max(0, int(math.floor(min(point[1] for point in projected)))),
        min(width - 1, int(math.ceil(max(point[0] for point in projected)))),
        min(height - 1, int(math.ceil(max(point[1] for point in projected)))),
    ]
    if box[0] > box[2] or box[1] > box[3]:
        return None
    return box


def find_actor_mask(raw_data, width, height, projected_box, expected_tags):
    if projected_box is None:
        return None
    x1, y1, x2, y2 = projected_box
    bgra = np.frombuffer(raw_data, dtype=np.uint8).reshape(height, width, 4)
    semantic_tags = bgra[y1 : y2 + 1, x1 : x2 + 1, 2]
    class_mask = np.isin(semantic_tags, list(expected_tags))
    local_y, local_x = np.where(class_mask)
    if local_y.size == 0:
        return None
    box = [
        int(x1 + local_x.min()),
        int(y1 + local_y.min()),
        int(x1 + local_x.max()),
        int(y1 + local_y.max()),
    ]
    projected_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    pixel_count = int(class_mask.sum())
    overlap = bbox_iou(projected_box, box)
    visible_fraction = pixel_count / float(projected_area)
    return {
        "pixel_count": pixel_count,
        "bbox": box,
        "iou": overlap,
        "visible_fraction": visible_fraction,
        "semantic_tags": sorted(expected_tags),
        "accepted": (
            overlap >= 0.1
            and box[3] - box[1] + 1 > 25
            and visible_fraction > 0.15
        ),
    }


def yolo_row(class_id, box, width, height):
    x1, y1, x2, y2 = box
    values = (
        (x1 + x2 + 1) / (2.0 * width),
        (y1 + y2 + 1) / (2.0 * height),
        (x2 - x1 + 1) / float(width),
        (y2 - y1 + 1) / float(height),
    )
    if not all(0.0 <= value <= 1.0 for value in values):
        raise ValueError("Non-normalized YOLO label: {}".format(values))
    return "{} {:.8f} {:.8f} {:.8f} {:.8f}".format(class_id, *values)


def weather_for(name):
    if name == "ClearNoon":
        return carla.WeatherParameters.ClearNoon
    if name == "MidRainySunset":
        return carla.WeatherParameters(
            cloudiness=70.0,
            precipitation=35.0,
            precipitation_deposits=25.0,
            wind_intensity=30.0,
            sun_azimuth_angle=180.0,
            sun_altitude_angle=10.0,
            wetness=45.0,
            fog_density=0.0,
        )
    if name == "HeavyRainNoon":
        return carla.WeatherParameters(
            cloudiness=95.0,
            precipitation=80.0,
            precipitation_deposits=70.0,
            wind_intensity=60.0,
            sun_azimuth_angle=45.0,
            sun_altitude_angle=75.0,
            wetness=90.0,
            fog_density=5.0,
        )
    raise ValueError("Unknown weather preset: {}".format(name))


def receive_frame(image_queue, frame_number):
    while True:
        image = image_queue.get(timeout=60)
        if image.frame == frame_number:
            return image
        if image.frame > frame_number:
            raise RuntimeError(
                "Camera skipped requested frame {} and delivered {}".format(
                    frame_number, image.frame
                )
            )


def spawn_nearby_actor(world, blueprint, ego_transform, local_x, local_y, yaw_offset=0):
    local_location = carla.Location(x=local_x, y=local_y, z=0.35)
    location = ego_transform.transform(local_location)
    transform = carla.Transform(
        location,
        carla.Rotation(
            pitch=0.0,
            yaw=ego_transform.rotation.yaw + yaw_offset,
            roll=0.0,
        ),
    )
    return world.try_spawn_actor(blueprint, transform)


def save_rgb(image, path):
    bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
        image.height, image.width, 4
    )
    rgb = bgra[:, :, 2::-1]
    Image.fromarray(rgb, mode="RGB").save(str(path), quality=95, subsampling=0)


def save_segmentation_debug(image, path):
    bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
        image.height, image.width, 4
    )
    Image.fromarray(bgra[:, :, 2::-1], mode="RGB").save(str(path))


def prepare_output(output):
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("label_policy_version") != "semantic-ignore-v1":
            raise ValueError(
                "Existing CARLA labels predate ignored-visible-object handling; "
                "generate into a new output directory"
            )
        return manifest
    return {
        "seed": 2026,
        "label_policy_version": "semantic-ignore-v1",
        "class_names": [
            "Pedestrian",
            "Cyclist",
            "IgnoreDontCare",
            "IgnorePersonSitting",
        ],
        "image_size": list(RGB_SIZE),
        "split_unit": "episode",
        "split_rule": "episode index modulo 5: 1/5 validation, 4/5 train",
        "weather_presets": {
            "ClearNoon": 0.5,
            "MidRainySunset": 0.3,
            "HeavyRainNoon": 0.2,
        },
        "episodes": [],
    }


def write_dataset_yaml(output):
    content = (
        "path: {}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: Pedestrian\n"
        "  1: Cyclist\n"
        "  2: IgnoreDontCare\n"
        "  3: IgnorePersonSitting\n"
    ).format(output.resolve().as_posix())
    (output / "dataset.yaml").write_text(content, encoding="utf-8")


def generate(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = prepare_output(output)
    town_prefix = "T01" if args.map == "Town01" else "T02"
    existing = {
        item["episode_id"]
        for item in manifest["episodes"]
    }
    existing_map_episodes = [
        item for item in manifest["episodes"] if item.get("map") == args.map
    ]
    expected_seed_offset = 10000 if args.map == "Town02" else 0
    for episode in existing_map_episodes:
        expected_episode_seed = args.seed + episode["start_spawn_point"] + expected_seed_offset
        if episode["seed"] != expected_episode_seed or (
            not args.allow_mixed_episode_lengths
            and episode["frames"] != args.frames_per_episode
        ):
            raise ValueError(
                "Existing episode configuration differs; use a new output directory "
                "or explicitly allow mixed episode lengths"
            )

    client = carla.Client(args.host, args.port)
    client.set_timeout(300.0)
    world = client.get_world()
    if not world.get_map().name.endswith(args.map):
        print("Loading map {} ...".format(args.map), flush=True)
        world = client.load_world(args.map)
        if not world.get_map().name.endswith(args.map):
            raise RuntimeError(
                "Expected map {}, got {}".format(args.map, world.get_map().name)
            )
    client.set_timeout(30.0)

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    traffic_manager = client.get_trafficmanager(args.traffic_manager_port)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_random_device_seed(args.seed)

    blueprint_library = world.get_blueprint_library()
    car_blueprints = blueprint_library.filter("vehicle.*")
    ego_blueprint = blueprint_library.find("vehicle.tesla.model3")
    bicycle_ids = (
        "vehicle.bh.crossbike",
        "vehicle.diamondback.century",
        "vehicle.gazelle.omafiets",
    )
    bicycle_blueprints = [blueprint_library.find(item) for item in bicycle_ids]
    walker_blueprints = list(blueprint_library.filter("walker.pedestrian.*"))
    if not bicycle_blueprints or not walker_blueprints:
        raise RuntimeError("CARLA is missing bicycle or pedestrian blueprints")

    spawn_points = list(world.get_map().get_spawn_points())
    if len(spawn_points) < args.episodes:
        raise ValueError(
            "Map has {} spawn points for {} requested episodes".format(
                len(spawn_points), args.episodes
            )
        )
    rng = random.Random(args.seed + (1 if args.map == "Town01" else 2))
    rng.shuffle(spawn_points)
    weather_order = (
        ["ClearNoon"] * 5 + ["MidRainySunset"] * 3 + ["HeavyRainNoon"] * 2
    )
    weather_schedule = [
        weather_order[index % len(weather_order)] for index in range(args.episodes)
    ]
    rng.shuffle(weather_schedule)

    added_episodes = []
    stop_episode = args.episodes if args.stop_episode is None else args.stop_episode
    for episode_index in range(args.start_episode, stop_episode):
        episode_id = "{}_E{:03d}".format(town_prefix, episode_index)
        if episode_id in existing:
            continue
        episode_rng = random.Random(args.seed + episode_index + (
            10000 if args.map == "Town02" else 0
        ))
        split = "val" if episode_index % 5 == 0 else "train"
        weather_name = weather_schedule[episode_index]
        print(
            "Starting {} ({}, {}, spawn point {})".format(
                episode_id, split, weather_name, episode_index
            ),
            flush=True,
        )
        world.set_weather(weather_for(weather_name))
        print("Weather set for {}".format(episode_id), flush=True)
        actors = []
        cameras = []
        frame_count = 0
        class_counts = Counter()
        skipped_count = 0
        ignored_visible_count = 0
        route_points = []
        try:
            ego = world.try_spawn_actor(ego_blueprint, spawn_points[episode_index])
            if ego is None:
                raise RuntimeError("Could not spawn episode ego vehicle")
            actors.append(ego)
            print("Ego spawned for {}".format(episode_id), flush=True)
            ego.set_simulate_physics(False)
            camera_transform = carla.Transform(
                carla.Location(x=1.5, z=1.6),
                carla.Rotation(pitch=-2.0),
            )
            camera_queues = {}
            for sensor_name, filename in (
                ("sensor.camera.rgb", "rgb"),
                ("sensor.camera.semantic_segmentation", "semantic"),
            ):
                blueprint = blueprint_library.find(sensor_name)
                blueprint.set_attribute("image_size_x", str(RGB_SIZE[0]))
                blueprint.set_attribute("image_size_y", str(RGB_SIZE[1]))
                blueprint.set_attribute("fov", "90")
                camera = world.spawn_actor(
                    blueprint, camera_transform, attach_to=ego
                )
                cameras.append(camera)
                image_queue = queue.Queue()
                camera.listen(image_queue.put)
                camera_queues[filename] = image_queue
            print("Cameras ready for {}".format(episode_id), flush=True)

            cyclists = []
            for cyclist_index in range(args.cyclists):
                bicycle_blueprint = bicycle_blueprints[
                    episode_rng.randrange(len(bicycle_blueprints))
                ]
                bicycle = spawn_nearby_actor(
                    world,
                    bicycle_blueprint,
                    spawn_points[episode_index],
                    14.0 + cyclist_index * 10.0,
                    (-1.6 if cyclist_index % 2 == 0 else 1.6),
                    episode_rng.uniform(-8.0, 8.0),
                )
                if bicycle is not None:
                    actors.append(bicycle)
                    bicycle.set_simulate_physics(False)
                    cyclists.append(bicycle)

            pedestrians = []
            for pedestrian_index in range(args.pedestrians):
                walker_blueprint = walker_blueprints[
                    episode_rng.randrange(len(walker_blueprints))
                ]
                walker = spawn_nearby_actor(
                    world,
                    walker_blueprint,
                    spawn_points[episode_index],
                    10.0 + pedestrian_index * 8.0,
                    (-4.2 if pedestrian_index % 2 == 0 else 4.2),
                    episode_rng.uniform(-25.0, 25.0),
                )
                if walker is not None:
                    actors.append(walker)
                    walker.set_simulate_physics(False)
                    pedestrians.append(walker)

            if not cyclists or not pedestrians:
                raise RuntimeError(
                    "Could not spawn both classes (cyclists={}, pedestrians={})".format(
                        len(cyclists), len(pedestrians)
                    )
                )
            print("Actors ready for {}".format(episode_id), flush=True)

            current_waypoint = world.get_map().get_waypoint(
                spawn_points[episode_index].location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if current_waypoint is None:
                raise RuntimeError("Episode start is not on a driving lane")
            print("Route ready for {}".format(episode_id), flush=True)

            for frame_index in range(args.frames_per_episode):
                route_options = current_waypoint.next(2.0)
                if route_options:
                    current_waypoint = route_options[
                        episode_rng.randrange(len(route_options))
                    ]
                route_transform = current_waypoint.transform
                route_points.append(
                    [
                        round(route_transform.location.x, 3),
                        round(route_transform.location.y, 3),
                        round(route_transform.rotation.yaw, 3),
                    ]
                )
                ego.set_transform(route_transform)
                for actor_index, bicycle in enumerate(cyclists):
                    local_x = 12.0 + actor_index * 12.0 + episode_rng.uniform(
                        -1.5, 1.5
                    )
                    local_y = episode_rng.uniform(-2.2, 2.2)
                    location = route_transform.transform(
                        carla.Location(x=local_x, y=local_y, z=0.3)
                    )
                    bicycle.set_transform(
                        carla.Transform(
                            location,
                            carla.Rotation(
                                yaw=route_transform.rotation.yaw
                                + episode_rng.uniform(-15.0, 15.0)
                            ),
                        )
                    )
                for actor_index, walker in enumerate(pedestrians):
                    local_x = 11.0 + actor_index * 9.0 + episode_rng.uniform(
                        -1.5, 1.5
                    )
                    local_y = (-4.0 if actor_index % 2 == 0 else 4.0)
                    local_y += episode_rng.uniform(-0.75, 0.75)
                    location = route_transform.transform(
                        carla.Location(x=local_x, y=local_y, z=0.35)
                    )
                    walker.set_transform(
                        carla.Transform(
                            location,
                            carla.Rotation(
                                yaw=route_transform.rotation.yaw
                                + episode_rng.uniform(-30.0, 30.0)
                            ),
                        )
                    )

                if args.debug_progress:
                    print(
                        "{} frame {} actors positioned".format(
                            episode_id, frame_index
                        ),
                        flush=True,
                    )
                frame_number = world.tick()
                if args.debug_progress:
                    print(
                        "{} frame {} tick {}".format(
                            episode_id, frame_index, frame_number
                        ),
                        flush=True,
                    )
                rgb_image = receive_frame(camera_queues["rgb"], frame_number)
                semantic_image = receive_frame(
                    camera_queues["semantic"], frame_number
                )
                if args.debug_progress:
                    print(
                        "{} frame {} images received".format(
                            episode_id, frame_index
                        ),
                        flush=True,
                    )
                if frame_index < args.save_segmentation_frames:
                    debug_dir = output / "debug_semantic"
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    save_segmentation_debug(
                        semantic_image,
                        debug_dir / "{}.png".format(episode_id),
                    )
                image_id = "{}_F{:06d}".format(episode_id, frame_index)
                image_path = output / "images" / split / (image_id + ".jpg")
                label_path = output / "labels" / split / (image_id + ".txt")
                save_rgb(rgb_image, image_path)

                rows = []
                debug_records = []
                for class_name, class_actors, required_tags in (
                    ("Cyclist", cyclists, {13, 19}),
                    ("Pedestrian", pedestrians, {12}),
                ):
                    for actor in class_actors:
                        projected = project_bbox(
                            actor,
                            cameras[0],
                            semantic_image.width,
                            semantic_image.height,
                            90.0,
                        )
                        mask = find_actor_mask(
                            semantic_image.raw_data,
                            semantic_image.width,
                            semantic_image.height,
                            projected,
                            required_tags,
                        )
                        if args.debug_matches and frame_index < 2:
                            debug_records.append(
                                {
                                    "class": class_name,
                                    "projected_bbox": projected,
                                    "mask": mask,
                                }
                            )
                        if mask is None:
                            skipped_count += 1
                            continue
                        if not mask["accepted"]:
                            skipped_count += 1
                            rows.append(
                                yolo_row(
                                    CLASS_IDS["IgnoreDontCare"],
                                    mask["bbox"],
                                    semantic_image.width,
                                    semantic_image.height,
                                )
                            )
                            ignored_visible_count += 1
                            continue
                        rows.append(
                            yolo_row(
                                CLASS_IDS[class_name],
                                mask["bbox"],
                                semantic_image.width,
                                semantic_image.height,
                            )
                        )
                        class_counts[class_name] += 1
                label_path.write_text(
                    "\n".join(rows) + ("\n" if rows else ""),
                    encoding="ascii",
                )
                if args.debug_progress:
                    print(
                        "{} frame {} saved".format(episode_id, frame_index),
                        flush=True,
                    )
                if args.debug_matches and frame_index < 2:
                    debug_path = output / "debug_matches.jsonl"
                    with debug_path.open("a", encoding="utf-8") as stream:
                        stream.write(
                            json.dumps(
                                {
                                    "episode_id": episode_id,
                                    "frame": frame_index,
                                    "objects": debug_records,
                                }
                            )
                            + "\n"
                        )
                frame_count += 1
                if frame_count % args.log_every == 0:
                    print(
                        "{} {}/{} frames; labels={}".format(
                            episode_id, frame_count, args.frames_per_episode,
                            dict(class_counts),
                        ),
                        flush=True,
                    )
        finally:
            for camera in cameras:
                camera.stop()
            for actor in reversed(actors):
                actor.destroy()
            world.tick()

        episode_record = {
            "episode_id": episode_id,
            "map": args.map,
            "weather": weather_name,
            "seed": args.seed + episode_index + (
                10000 if args.map == "Town02" else 0
            ),
            "split": split,
            "start_spawn_point": episode_index,
            "frames": frame_count,
            "class_instance_counts": dict(sorted(class_counts.items())),
            "filtered_instances": skipped_count,
            "ignored_visible_instances": ignored_visible_count,
            "route_points_xy_yaw": route_points,
        }
        manifest["episodes"].append(episode_record)
        added_episodes.append(episode_record)
        print(
            "Finished {} ({}, {} frames, labels={})".format(
                episode_id, split, frame_count, dict(class_counts)
            ),
            flush=True,
        )
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    manifest.setdefault("generation_config", {})[args.map] = {
        "carla_server": client.get_server_version(),
        "python_client": client.get_client_version(),
        "camera": {
            "sensor": "sensor.camera.rgb + sensor.camera.semantic_segmentation",
            "width": RGB_SIZE[0],
            "height": RGB_SIZE[1],
            "horizontal_fov_degrees": 90,
            "fixed_delta_seconds": 0.05,
        },
        "map": args.map,
        "generator_seed": args.seed,
        "episodes_per_map": args.episodes,
        "frames_per_episode": args.frames_per_episode,
        "cyclists_per_episode": args.cyclists,
        "pedestrians_per_episode": args.pedestrians,
        "visibility_filter": (
            "project actor 3D bounding-box vertices; clip expected semantic tags to "
            "the resulting image rectangle; accept if bbox IoU >= 0.1, mask height > "
            "25 px, and semantic-pixel fraction of projected rectangle > 0.15"
        ),
        "rejected_visible_object_policy": (
            "if expected semantic pixels remain but acceptance thresholds fail, "
            "write their visible-pixel bounding box as IgnoreDontCare so it is "
            "excluded from background classification loss; candidates with no "
            "matching semantic pixels produce no ignore box"
        ),
        "class_mapping": {
            "cyclist": "vehicle.bh.crossbike, vehicle.diamondback.century, vehicle.gazelle.omafiets actor type",
            "pedestrian": "walker.pedestrian.* actor type",
        },
        "semantic_mask": "semantic tags clipped to projected actor 3D bounding box; Pedestrians=12, Rider=13, Bicycle=19",
        "rgb_encoding": "JPEG quality 95, subsampling 0",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_dataset_yaml(output)
    world.apply_settings(original_settings)
    traffic_manager.set_synchronous_mode(False)
    print(
        json.dumps(
            {
                "added_episodes": len(added_episodes),
                "frames": sum(item["frames"] for item in added_episodes),
                "output": str(output),
            },
            indent=2,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--traffic-manager-port", type=int, default=8000)
    parser.add_argument("--map", choices=("Town01", "Town02"), required=True)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--stop-episode", type=int)
    parser.add_argument("--frames-per-episode", type=int, default=188)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--debug-progress", action="store_true")
    parser.add_argument("--allow-mixed-episode-lengths", action="store_true")
    parser.add_argument("--cyclists", type=int, default=2)
    parser.add_argument("--pedestrians", type=int, default=2)
    parser.add_argument("--save-segmentation-frames", type=int, default=0)
    parser.add_argument("--debug-matches", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "data" / "carla_yolo",
    )
    args = parser.parse_args()
    if args.episodes <= 0 or args.frames_per_episode <= 0:
        parser.error("episodes and frames-per-episode must be positive")
    if args.log_every <= 0:
        parser.error("log-every must be positive")
    if args.start_episode < 0 or args.start_episode >= args.episodes:
        parser.error("start-episode must be within the planned episode range")
    if args.stop_episode is not None and not (
        args.start_episode < args.stop_episode <= args.episodes
    ):
        parser.error("stop-episode must be greater than start and <= episodes")
    if args.cyclists <= 0 or args.pedestrians <= 0:
        parser.error("at least one cyclist and pedestrian are required")
    generate(args)


if __name__ == "__main__":
    main()
