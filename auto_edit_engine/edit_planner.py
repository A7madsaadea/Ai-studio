
"""Build an initial edit plan from reference, media, and matching analysis."""

import json
import sys
from pathlib import Path


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def resolve_path(project_root, value):
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return str(path.resolve())



def sync_scene_durations(scenes, audio_data, tolerance=0.25, min_duration=0.25):
    """Prefer musical beats, using onsets only as a fallback."""
    if not scenes:
        return {}, 0

    boundaries = [float(scenes[0].get("start", 0) or 0)]

    for scene in scenes[1:]:
        boundaries.append(float(scene.get("start", 0) or 0))

    last = scenes[-1]
    end_time = float(
        last.get("end", 0)
        or (
            float(last.get("start", 0) or 0)
            + float(last.get("duration", 0) or 0)
        )
    )
    boundaries.append(end_time)

    if not audio_data.get("available"):
        return {
            scene.get("index", i): round(
                float(scene.get("duration", 0) or 0), 3
            )
            for i, scene in enumerate(scenes)
        }, 0

    beats = sorted(float(t) for t in audio_data.get("beats", []))
    onsets = sorted(float(t) for t in audio_data.get("onsets", []))

    adjusted = [boundaries[0]]
    synced_count = 0

    for i in range(1, len(boundaries) - 1):
        original = boundaries[i]
        lower = adjusted[-1] + min_duration
        upper = boundaries[i + 1] - min_duration

        beat_candidates = [
            t for t in beats
            if lower <= t <= upper
            and abs(t - original) <= tolerance
        ]

        onset_candidates = [
            t for t in onsets
            if lower <= t <= upper
            and abs(t - original) <= tolerance
        ]

        candidates = beat_candidates or onset_candidates

        if candidates:
            chosen = min(candidates, key=lambda t: abs(t - original))
            if abs(chosen - original) > 0.001:
                synced_count += 1
        else:
            chosen = original

        adjusted.append(chosen)

    adjusted.append(end_time)

    durations = {
        scene.get("index", i): round(
            max(0.001, adjusted[i + 1] - adjusted[i]), 3
        )
        for i, scene in enumerate(scenes)
    }

    return durations, synced_count


def build_plan(analysis_dir):
    analysis_dir = Path(analysis_dir).resolve()
    project_root = analysis_dir.parent

    reference = load_json(analysis_dir / "reference.json")
    media_data = load_json(analysis_dir / "media.json")
    matches_data = load_json(analysis_dir / "matches.json")

    audio_path = analysis_dir / "audio.json"
    audio_data = (
        load_json(audio_path)
        if audio_path.exists()
        else {"available": False, "cut_points": []}
    )

    scenes = reference.get("scenes", [])
    synced_durations, synced_count = sync_scene_durations(
        scenes, audio_data
    )


    media_by_id = {
        item["id"]: item
        for item in media_data.get("items", [])
        if item.get("id")
    }
    matches_by_scene = {
        item["scene_index"]: item
        for item in matches_data.get("matches", [])
        if "scene_index" in item
    }

    reference_video = reference.get("video", {})
    orientation = reference_video.get("orientation", "vertical")

    if orientation == "horizontal":
        width, height = 1920, 1080
    else:
        width, height = 1080, 1920

    timeline = []
    warnings = []
    timeline_cursor = 0.0

    for scene in scenes:
        scene_index = scene.get("index")
        target_duration = float(
            synced_durations.get(
                scene_index,
                scene.get("duration", 0)
            ) or 0
        )

        if target_duration <= 0:
            warnings.append(
                f"Scene {scene_index}: invalid duration; skipped."
            )
            continue

        match = matches_by_scene.get(scene_index, {})
        chosen = match.get("chosen")

        if not chosen:
            warnings.append(
                f"Scene {scene_index}: no matched media; skipped."
            )
            continue

        media_id = chosen.get("media_id")
        media = media_by_id.get(media_id)

        if not media:
            warnings.append(
                f"Scene {scene_index}: media {media_id!r} "
                "was not found in media.json; skipped."
            )
            continue

        if media.get("status") != "ok":
            warnings.append(
                f"Scene {scene_index}: media {media_id} "
                "is not marked as OK; skipped."
            )
            continue

        media_type = media.get("type")
        source_duration = media.get("duration")
        source_in = max(
            0.0,
            float(chosen.get("source_in", media.get("best_start", 0)) or 0)
        )

        fill_mode = "none"

        if media_type == "image":
            source_in = 0.0
            source_out = None
            actual_source_duration = None

        elif media_type == "video":
            if source_duration is None or float(source_duration) <= 0:
                warnings.append(
                    f"Scene {scene_index}: invalid source duration "
                    f"for {media_id}; skipped."
                )
                continue

            source_duration = float(source_duration)

            # If the selected start leaves insufficient footage, move
            # backward to fit the complete target duration when possible.
            if source_in + target_duration > source_duration:
                if source_duration >= target_duration:
                    source_in = source_duration - target_duration
                    warnings.append(
                        f"Scene {scene_index}: source start adjusted "
                        f"to {source_in:.3f}s to fit the scene."
                    )
                else:
                    source_in = 0.0
                    fill_mode = "freeze_last_frame"
                    warnings.append(
                        f"Scene {scene_index}: source is shorter than "
                        "the target scene; the renderer must freeze "
                        "the final frame to fill the remaining time."
                    )

            source_out = min(
                source_duration,
                source_in + target_duration
            )
            actual_source_duration = source_out - source_in

        else:
            warnings.append(
                f"Scene {scene_index}: unsupported media type "
                f"{media_type!r}; skipped."
            )
            continue

        clip = {
            "scene_index": scene_index,
            "timeline_start": round(timeline_cursor, 3),
            "timeline_end": round(timeline_cursor + target_duration, 3),
            "duration": round(target_duration, 3),
            "source": {
                "media_id": media_id,
                "name": media.get("name", chosen.get("name", "")),
                "type": media_type,
                "path": media.get("path"),
                "resolved_path": resolve_path(
                    project_root, media["path"]
                ),
                "source_in": round(source_in, 3),
                "source_out": (
                    round(source_out, 3)
                    if source_out is not None
                    else None
                ),
                "available_duration": (
                    round(actual_source_duration, 3)
                    if actual_source_duration is not None
                    else None
                ),
            },
            "match": {
                "score": chosen.get("score"),
                "similarity": chosen.get("similarity"),
                "orientation_ok": chosen.get("orientation_ok"),
                "match_frame_t": chosen.get("match_frame_t"),
            },
            "transition_in": scene.get(
                "transition_in", {"type": "cut"}
            ),
            "fill_mode": fill_mode,
        }

        timeline.append(clip)
        timeline_cursor += target_duration

    plan = {
        "version": 1,
        "status": "planned" if timeline else "empty",
        "reference": {
            "path": str(analysis_dir / "reference.json"),
            "duration": reference_video.get("duration"),
            "orientation": orientation,
        },
        "output": {
            "width": width,
            "height": height,
            "fps": reference_video.get("fps", 30),
            "duration": round(timeline_cursor, 3),
            "format": "mp4",
        },
        "summary": {
            "reference_scenes": len(reference.get("scenes", [])),
            "planned_clips": len(timeline),
            "warnings": len(warnings),
        },
        "clips": timeline,
        "warnings": warnings,
    }

    output_path = analysis_dir / "edit_plan.json"
    save_json(output_path, plan)

    print(json.dumps(plan["summary"], ensure_ascii=False, indent=2))
    print(f"Plan saved to: {output_path}")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"- {warning}")

    print(f"Beat-synced boundaries: {synced_count}")
    return plan

if __name__ == "__main__":
    analysis_folder = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path("test_out")
    )
    build_plan(analysis_folder)
