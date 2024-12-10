import argparse
import matplotlib.pyplot as plt
from colorizers import *
from colorizers.generator import Generator
from colorizers.data_loader import get_data_loader

parser = argparse.ArgumentParser()
parser.add_argument(
    "--use_gpu", action="store_true", help="whether to use GPU"
)
opt = parser.parse_args()

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

val_loader = get_data_loader(
    batch_size=1, root_dir='/home/jdanny/colorization-442/data/landscape_images/val', HW=(256,256)
)

# load colorizers
colorizer_eccv16 = eccv16()
if opt.use_gpu:
    colorizer_eccv16.cuda()

colorizer_eccv16 = colorizer_eccv16.to(device)

gen = Generator()  # Instantiate the model
checkpoint = torch.load('/home/jdanny/colorization-442/checkpoints/MSE_checkpoint.pt', map_location=device)
gen.load_state_dict(checkpoint["generator_state_dict"])
gen.to(device)  # Move the model to the correct device
gen.eval()
colorizer_eccv16.eval()

criterion_mse = torch.nn.MSELoss()
# ---------------------
# Validation Phase
# ---------------------
val_loss = 0
with torch.no_grad():  # Disable gradient computation
    for val_batch in val_loader:
        val_l_rs = val_batch["L_resized"].to(device)
        val_ab_rs = val_batch["AB_resized"].to(device)
        val_pred_ab = colorizer_eccv16(val_l_rs)
        val_pred_ab = torch.clamp(val_pred_ab / 128.0, -1, 1)
        val_loss += criterion_mse(val_pred_ab, val_ab_rs).item()
        # print("colorizer_eccv16 output range:", val_pred_ab.min().item(), val_pred_ab.max().item())


val_loss /= len(val_loader)
print(f"Orig Validation Loss: {val_loss:.4f}")

val_loss = 0
with torch.no_grad():  # Disable gradient computation
    for val_batch in val_loader:
        val_l_rs = val_batch["L_resized"].to(device)
        val_ab_rs = val_batch["AB_resized"].to(device)
        val_pred_ab = gen(val_l_rs)
        val_loss += criterion_mse(val_pred_ab, val_ab_rs).item()
        # print("gen output range:", val_pred_ab.min().item(), val_pred_ab.max().item())
        # print("Ground truth range:", val_ab_rs.min().item(), val_ab_rs.max().item())


val_loss /= len(val_loader)
print(f"Gen Validation Loss: {val_loss:.4f}")