import argparse
import hashlib
import json
from pathlib import Path


def assign_episode_splits(episodes, train_max_x=303.0, val_min_x=323.0):
    if val_min_x - train_max_x < 20.0:
        raise ValueError("Train and validation regions need at least 20 m of separation")

    split_by_episode = {}
    excluded_by_map = {}
    for episode in episodes:
        episode_id = episode["episode_id"]
        map_name = episode["map"]
        points = episode.get("route_points_xy_yaw", [])
        if not points:
            raise ValueError("Episode has no route points: {}".format(episode_id))

        if map_name == "Town01":
            minimum_x = min(point[0] for point in points)
            maximum_x = max(point[0] for point in points)
            if maximum_x <= train_max_x:
                split_by_episode[episode_id] = "train"
            elif minimum_x >= val_min_x:
                split_by_episode[episode_id] = "val"
            else:
                excluded_by_map.setdefault(map_name, []).append(episode_id)
        elif map_name == "Town02":
            split_by_episode[episode_id] = "val"
        else:
            raise ValueError("Unexpected CARLA map: {}".format(map_name))

    if not any(split == "train" for split in split_by_episode.values()):
        raise ValueError("Region assignment produced no training episodes")
    if not any(split == "val" for split in split_by_episode.values()):
        raise ValueError("Region assignment produced no validation episodes")
    return split_by_episode, excluded_by_map


def prepare_region_split(source, output, train_max_x=303.0, val_min_x=323.0):
    source = Path(source).resolve()
    output = Path(output).resolve()
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite non-empty dataset: {}".format(output))

    source_manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(source_manifest_bytes.decode("utf-8"))
    split_by_episode, excluded_by_map = assign_episode_splits(
        manifest["episodes"], train_max_x, val_min_x
    )
    selected_episodes = []
    for episode in manifest["episodes"]:
        split = split_by_episode.get(episode["episode_id"])
        if split is None:
            continue
        selected = dict(episode)
        selected["split"] = split
        selected_episodes.append(selected)

    if output == source or source in output.parents:
        raise ValueError("Output must not replace or be nested inside the source dataset")
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    expected_frames = {
        episode["episode_id"]: episode["frames"] for episode in selected_episodes
    }
    copied_frames = {episode_id: 0 for episode_id in expected_frames}
    for source_split in ("train", "val"):
        image_dir = source / "images" / source_split
        label_dir = source / "labels" / source_split
        for image_path in image_dir.iterdir():
            if image_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            image_id = image_path.stem
            episode_id = image_id.rsplit("_F", 1)[0]
            destination_split = split_by_episode.get(episode_id)
            if destination_split is None:
                continue
            label_path = label_dir / (image_id + ".txt")
            if not label_path.is_file():
                raise FileNotFoundError(label_path)
            image_target = output / "images" / destination_split / image_path.name
            label_target = output / "labels" / destination_split / label_path.name
            image_target.hardlink_to(image_path)
            label_target.hardlink_to(label_path)
            copied_frames[episode_id] += 1

    incomplete = {
        episode_id: (copied_frames[episode_id], frames)
        for episode_id, frames in expected_frames.items()
        if copied_frames[episode_id] != frames
    }
    if incomplete:
        raise ValueError("Selected episode image counts do not match manifest: {}".format(incomplete))

    manifest["episodes"] = selected_episodes
    manifest["split_rule"] = (
        "Town01 routes wholly at x <= {:.3f} m are train; routes wholly at "
        "x >= {:.3f} m are validation; Town02 is held out for validation; "
        "routes in the Town01 buffer are excluded"
    ).format(train_max_x, val_min_x)
    manifest["split_selection"] = {
        "strategy": (
            "Town01 complete routes assigned by X-region; Town02 held out for validation; "
            "episodes crossing the Town01 buffer are excluded"
        ),
        "source_manifest_sha256": hashlib.sha256(source_manifest_bytes).hexdigest(),
        "source_dataset": str(source),
        "train_max_x_m": train_max_x,
        "validation_min_x_m": val_min_x,
        "minimum_axis_buffer_m": val_min_x - train_max_x,
        "excluded_episode_ids_by_map": {
            map_name: sorted(ids) for map_name, ids in excluded_by_map.items()
        },
        "included_episode_ids_by_split": {
            split: sorted(
                episode_id
                for episode_id, assigned_split in split_by_episode.items()
                if assigned_split == split
            )
            for split in ("train", "val")
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    class_names = manifest["class_names"]
    yaml_lines = [
        "path: {}".format(output.as_posix()),
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    yaml_lines.extend(
        "  {}: {}".format(index, name) for index, name in enumerate(class_names)
    )
    (output / "dataset.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    return {
        "output": str(output),
        "episode_counts": {
            split: sum(episode["split"] == split for episode in selected_episodes)
            for split in ("train", "val")
        },
        "frame_counts": {
            split: sum(
                episode["frames"]
                for episode in selected_episodes
                if episode["split"] == split
            )
            for split in ("train", "val")
        },
        "excluded_episode_ids_by_map": manifest["split_selection"][
            "excluded_episode_ids_by_map"
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-max-x", type=float, default=303.0)
    parser.add_argument("--val-min-x", type=float, default=323.0)
    args = parser.parse_args()
    result = prepare_region_split(
        args.source, args.output, args.train_max_x, args.val_min_x
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
