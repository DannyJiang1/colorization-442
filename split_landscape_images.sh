# Define source and destination directories
src_dir="/home/jdanny/colorization-442/data/landscape_images/train"
dest_dir="/home/jdanny/colorization-442/data/landscape_images/val"

# Move the first 80% of files
for i in $(seq 5702 7128); do
    mv "$src_dir/$i.jpg" "$dest_dir/"
done