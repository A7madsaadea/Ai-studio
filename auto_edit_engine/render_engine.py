
"""Render edit_plan.json into an MP4 video using FFmpeg."""

import json
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(command):
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(
            "FFmpeg command failed:\n"
            + result.stderr[-5000:]
        )
    return result


def render(analysis_dir, audio_path=None):
    analysis_dir = Path(analysis_dir).resolve()
    project_root = analysis_dir.parent
    plan_path = analysis_dir / "edit_plan.json"

    if not plan_path.exists():
        raise FileNotFoundError(f"Edit plan not found: {plan_path}")

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg was not found on PATH.")

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    clips = plan.get("clips", [])
    output_settings = plan.get("output", {})

    if not clips:
        raise RuntimeError("The edit plan contains no clips.")

    width = int(output_settings.get("width", 1920))
    height = int(output_settings.get("height", 1080))
    fps = float(output_settings.get("fps", 30))
    total_duration = float(output_settings.get("duration", 0))

    if total_duration <= 0 or fps <= 0:
        raise RuntimeError("Invalid output duration or FPS.")

    temp_dir = analysis_dir / "render_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    output_path = analysis_dir / "final_video.mp4"
    silent_path = temp_dir / "video_silent.mp4"
    concat_path = temp_dir / "concat_list.txt"

    segment_paths = []

    for index, clip in enumerate(clips):
        source = clip["source"]
        source_path = Path(source["resolved_path"])
        if not source_path.is_absolute():
            source_path = (project_root / source_path).resolve()

        if not source_path.exists():
            raise FileNotFoundError(
                f"Source for scene {clip.get('scene_index')} "
                f"does not exist: {source_path}"
            )

        duration = float(clip["duration"])
        source_type = source["type"]
        segment_path = temp_dir / f"clip_{index:03d}.mp4"

        filters = [
            f"scale={width}:{height}:force_original_aspect_ratio=increase",
            f"crop={width}:{height}",
            "setsar=1",
            f"fps={fps}",
            "setpts=PTS-STARTPTS",
        ]

        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]

        if source_type == "image":
            command += [
                "-loop", "1",
                "-framerate", str(fps),
                "-i", str(source_path),
            ]
            filters.insert(-1, f"trim=duration={duration}")

        elif source_type == "video":
            source_in = max(0.0, float(source.get("source_in", 0)))
            available = source.get("available_duration")
            available = (
                float(available)
                if available is not None
                else duration
            )
            available = max(0.0, min(available, duration))

            command += [
                "-ss", str(source_in),
                "-i", str(source_path),
            ]

            # Trim available footage, then freeze the final frame if needed.
            filters.insert(0, f"trim=duration={max(available, 0.04)}")

            missing = max(0.0, duration - available)
            if missing > 0.01:
                filters.append(
                    f"tpad=stop_mode=clone:stop_duration={missing}"
                )

            filters.append(f"trim=duration={duration}")

        else:
            raise RuntimeError(
                f"Unsupported media type: {source_type}"
            )

        # Add subtle zoom motion, alternating zoom-in and zoom-out.
        zoom_rate = 0.04 / max(duration * fps, 1)

        if int(clip.get("scene_index", 0)) % 2 == 0:
            zoom_expr = f"min(1+on*{zoom_rate:.8f},1.04)"
        else:
            zoom_expr = f"max(1.04-on*{zoom_rate:.8f},1.0)"

        filters.append(
            f"zoompan=z='{zoom_expr}':"
            f"x='iw/2-(iw/zoom/2)':"
            f"y='ih/2-(ih/zoom/2)':"
            f"d=1:s={width}x{height}:fps={fps}"
        )

        # Apply a brief white flash only when the reference labels the scene as flash.
        transition = clip.get("transition_in", {})
        if transition.get("type") == "flash":
            filters.append(
                "drawbox=x=0:y=0:w=iw:h=ih:"
                "color=white@0.85:t=fill:enable='lt(t,0.06)'"
            )

        # Normalize timestamps after all effects.
        filters.append("setpts=PTS-STARTPTS")


        command += [
            "-vf", ",".join(filters),
            "-an",
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-r", str(fps),
            "-movflags", "+faststart",
            str(segment_path),
        ]

        print(
            f"[{index + 1}/{len(clips)}] "
            f"Rendering scene {clip.get('scene_index')} "
            f"({duration:.2f}s): {source_path.name}"
        )
        run_command(command)

        if not segment_path.exists() or segment_path.stat().st_size == 0:
            raise RuntimeError(f"Segment render failed: {segment_path}")

        segment_paths.append(segment_path)

    # FFmpeg concat demuxer joins the normalized segments.
    def ffconcat_quote(path):
        value = path.resolve().as_posix()
        return "'" + value.replace("'", "'\\''") + "'"

    concat_text = "".join(
        f"file {ffconcat_quote(path)}\n"
        for path in segment_paths
    )
    concat_path.write_text(concat_text, encoding="utf-8")

    concat_command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_path),
        "-c", "copy",
        "-movflags", "+faststart",
        str(silent_path),
    ]
    print("Joining clips...")
    run_command(concat_command)

    if audio_path:
        audio_path = Path(audio_path).resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio source not found: {audio_path}")

        mux_command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(silent_path),
            "-i", str(audio_path),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-t", str(total_duration),
            "-movflags", "+faststart",
            str(output_path),
        ]
        print("Adding soundtrack...")
        run_command(mux_command)
    else:
        shutil.copy2(silent_path, output_path)

    print("\nRender complete!")
    print(f"Output: {output_path}")
    print(f"Expected duration: {total_duration:.2f}s")
    print(f"Resolution: {width}x{height} @ {fps:g} FPS")
    print(f"Size: {output_path.stat().st_size / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("test_out")
    soundtrack = None

    if len(sys.argv) > 2:
        soundtrack = Path(sys.argv[2])

    try:
        render(folder, soundtrack)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
