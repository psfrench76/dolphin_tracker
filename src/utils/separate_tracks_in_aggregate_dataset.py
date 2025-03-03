"""
This script separates track ID spaces in an aggregate dataset containing multiple video clips.
Each video clip should have its own track ID space to avoid conflicts. The script processes track files
in the specified directory, ensuring that track IDs are unique across different video clips.

The script performs the following steps:
1. Reads track files from the specified directory.
2. Assigns a unique clip ID to each video clip, incrementing the clip ID when encountering a new video name.
3. Creates a clip key in the format <clip_id>_<video_name>.
4. Optionally separates non-contiguous frames into separate clips if the --separate_non_contiguous flag is set.
5. Loads tracks from each file and updates statistics for each clip.
6. Increments track IDs to ensure uniqueness across clips.
7. Prints a summary of the processed clips.

Usage: separate_tracks_in_aggregate_dataset.py <tracks_dir> [--separate_non_contiguous]
"""

import argparse
from pathlib import Path
from data_conversion import increment_track_file_by_amount
from settings import settings

def process_tracks(tracks_dir, separate_non_contiguous):
    tracks_path = Path(tracks_dir)
    if tracks_path.name != settings['tracks_dir']:
        raise ValueError(f"The folder name must be '{settings['tracks_dir']}'.")

    track_files = sorted(tracks_path.glob('*.txt'))
    clip_stats = {}
    max_track_id = 0
    clip_id = 0
    last_video_name = None

    for track_file in track_files:
        video_name, frame_id = track_file.stem.rsplit('_', 1)
        frame_id = int(frame_id)

        if video_name != last_video_name:
            clip_id += 1
            last_video_name = video_name

        clip_key = f"{clip_id}_{video_name}"

        if separate_non_contiguous and clip_key in clip_stats:
            last_frame_id = clip_stats[clip_key]['last_frame_id']
            if frame_id != last_frame_id + 1 and not is_consistent_sparse_annotation(clip_stats[clip_key], frame_id):
                clip_id += 1
                clip_key = f"{clip_id}_{video_name}"

        if clip_key not in clip_stats:
            clip_stats[clip_key] = {
                'frames': [],
                'frame_ids': [],
                'original_min': float('inf'),
                'original_max': float('-inf'),
                'increment': 0,
                'new_min': float('inf'),
                'new_max': float('-inf'),
                'background_frames': 0,
                'frames_with_tracks': 0,
                'last_frame_id': frame_id,
                'annotation_interval': 1
            }

        tracks = load_tracks(track_file)
        if not tracks:
            clip_stats[clip_key]['background_frames'] += 1
        else:
            clip_stats[clip_key]['frames_with_tracks'] += 1

        original_min = min(tracks) if tracks else float('inf')
        original_max = max(tracks) if tracks else float('-inf')

        clip_stats[clip_key]['frames'].append(track_file)
        clip_stats[clip_key]['frame_ids'].append(frame_id)
        clip_stats[clip_key]['original_min'] = min(clip_stats[clip_key]['original_min'], original_min)
        clip_stats[clip_key]['original_max'] = max(clip_stats[clip_key]['original_max'], original_max)
        clip_stats[clip_key]['last_frame_id'] = frame_id

    for clip_key, stats in sorted(clip_stats.items(), key=lambda x: int(x[0].split('_')[0])):
        if stats['original_min'] <= max_track_id:
            increment = max_track_id - stats['original_min'] + 1
            for track_file in stats['frames']:
                increment_track_file_by_amount(track_file, increment)
            stats['increment'] = increment

        for track_file in stats['frames']:
            new_tracks = load_tracks(track_file)
            new_min = min(new_tracks) if new_tracks else float('inf')
            new_max = max(new_tracks) if new_tracks else float('-inf')

            stats['new_min'] = min(stats['new_min'], new_min)
            stats['new_max'] = max(stats['new_max'], new_max)

        max_track_id = max(max_track_id, stats['new_max'])

    print_summary(clip_stats, separate_non_contiguous)

def load_tracks(track_file_path):
    with open(track_file_path, 'r') as file:
        return [int(line.strip()) for line in file if line.strip()]

def is_consistent_sparse_annotation(stats, new_frame_id):
    frame_ids = stats['frame_ids']
    if len(frame_ids) < 2:
        if len(frame_ids) == 1 and (new_frame_id - frame_ids[-1]) <= 30:
            stats['annotation_interval'] = new_frame_id - frame_ids[-1]
            return True
        return False

    interval = stats['annotation_interval']
    return (new_frame_id - frame_ids[-1]) == interval

def print_summary(clip_stats, separate_non_contiguous):
    for clip_key, stats in sorted(clip_stats.items(), key=lambda x: int(x[0].split('_')[0])):
        print(f"Clip: {clip_key}")
        print(f"  Number of frames: {len(stats['frames'])}")
        print(f"  Frames with tracks: {stats['frames_with_tracks']}")
        print(f"  Background frames: {stats['background_frames']}")
        print(f"  Original track ID space: {stats['original_min']} - {stats['original_max']}")
        print(f"  Increment value: {stats['increment']}")
        print(f"  New track ID space: {stats['new_min']} - {stats['new_max']}")
        print(f"  Annotation interval: {stats['annotation_interval']}")
        if separate_non_contiguous:
            print(f"  Starting frame ID: {min(stats['frame_ids'])}")
            print(f"  Ending frame ID: {max(stats['frame_ids'])}")
        print()

    total_frames = sum(len(stats['frames']) for stats in clip_stats.values())
    frames_with_tracks = sum(stats['frames_with_tracks'] for stats in clip_stats.values())
    background_frames = sum(stats['background_frames'] for stats in clip_stats.values())

    print(f"Total frames: {total_frames}")
    print(f"Frames with tracks: {frames_with_tracks}")
    print(f"Background frames: {background_frames}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Separate track ID spaces in an aggregate dataset.")
    parser.add_argument('tracks_dir', type=str, help='Path to the tracks directory')
    parser.add_argument('--separate_non_contiguous', action='store_true', help='Separate non-contiguous frames into separate clips')
    args = parser.parse_args()

    process_tracks(args.tracks_dir, args.separate_non_contiguous)