import re
import numpy as np
from scipy.cluster.vq import kmeans, vq

def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple."""
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb_color):
    """Convert RGB tuple to hex color."""
    return "#{:02x}{:02x}{:02x}".format(*rgb_color)

def main():
    print("Enter 6-digit hex color codes one-by-one (e.g., ff5733). Type 'done' when finished:")
    hex_colors = []
    while True:
        user_input = input("Enter color: ").strip().lower()
        if user_input == "done":
            break
        if re.fullmatch(r"[0-9a-f]{6}", user_input):
            hex_colors.append(user_input)
        else:
            print("Invalid input. Please enter a valid 6-digit hex color code.")

    if not hex_colors:
        print("No colors entered. Exiting.")
        return

    rgb_colors = np.array([hex_to_rgb(color) for color in hex_colors])

    while True:
        try:
            k = int(input("Enter the number of clusters (k): "))
            if k > 0:
                break
            else:
                print("Please enter a positive integer.")
        except ValueError:
            print("Invalid input. Please enter a positive integer.")

    # Perform K-means clustering using scipy
    cluster_centers, _ = kmeans(rgb_colors.astype(float), k)
    labels, _ = vq(rgb_colors, cluster_centers)

    print("\nK-means clustering results:")
    for cluster_idx in range(k):
        cluster_mean = tuple(map(int, cluster_centers[cluster_idx]))
        print(f"\nGroup {cluster_idx + 1} (Mean: {rgb_to_hex(cluster_mean)}):")
        group_members = [hex_colors[i] for i in range(len(hex_colors)) if labels[i] == cluster_idx]
        for member in group_members:
            print(f"  {member}")

if __name__ == "__main__":
    main()