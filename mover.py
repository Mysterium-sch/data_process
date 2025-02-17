from pathlib import Path

main_dir = Path("")
data = Path("")


for subdir in main_dir.iterdir():
    images = Path(subdir, "processed", "images")
    labels = Path(subdir, "processed", "labels")
    lidar = Path(subdir, "processed", "lidar")
    for im in images.iterdir():
        lid_og = Path(lidar, (im.with_suffix('.npy')).name)
        la_og = Path(labels, (im.with_suffix('.txt')).name)

        im_dest = Path(data, "images", im.name)
        lid_dest = Path(data, "lidar", (im.with_suffix('.npy')).name)
        la_dest = Path(data, "labels", (im.with_suffix('.txt')).name)

        if(im.exists() and lid_og.exists() and la_og.exists):
            im.rename(im_dest)
            lid_og.rename(lid_dest)
            la_og.rename(la_dest)
            

    