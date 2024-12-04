import torch
from torch.optim import Adam
from tqdm import tqdm
from .data_loader import get_data_loader
from .eccv16 import ECCVGenerator
from .discriminator import discriminator
from .util import postprocess_tens, preprocess_img, load_img
from PIL import Image
import os
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def visualize_hardcoded_image(
    generator, image_path, HW, epoch, device, save_path=None
):
    """
    Visualizes the prediction for a hardcoded grayscale image.
    Args:
        generator (nn.Module): The trained generator model.
        image_path (str): Path to the hardcoded grayscale image (L channel).
        HW (tuple): Target image size (H, W).
        epoch (int): Current epoch (for labeling and saving).
        device (torch.device): Device to use for inference.
        save_path (str, optional): Directory to save the visualization.
    """
    # Ensure the model is in evaluation mode
    generator.eval()

    img = load_img(image_path)
    tens_orig_l, tens_resized_l = preprocess_img(img, HW=(256, 256))

    # Move tensors to the device
    tens_resized_l = tens_resized_l.to(device)

    # Generate predictions
    with torch.no_grad():
        pred_ab = generator(tens_resized_l)  # Predict AB channels

    # Unnormalize AB channels for visualization
    pred_ab = pred_ab * 128  # Convert from [-1, 1] to [-128, 127]

    # Postprocess to combine L and AB channels and convert to RGB
    pred_rgb = postprocess_tens(tens_resized_l, pred_ab)

    # Plot and save the visualization
    plt.figure(figsize=(8, 4))
    plt.title(f"Epoch {epoch} - Predicted Colorization")
    plt.imshow(pred_rgb)
    plt.axis("off")

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/epoch_{epoch}_hardcoded_visualization.png")
        print(
            f"Saved hardcoded visualization for epoch {epoch} to {save_path}/epoch_{epoch}_hardcoded_visualization.png"
        )

    # Set the model back to training mode
    generator.train()


def train_disc(generator, disc, tens_l_rs, tens_ab_rs, optimizer_disc, device):
    # Generate fake colorizations
    pred_ab = generator(tens_l_rs).detach()
    # Concatenate L and AB channels to form 6-channel input
    fake_img = torch.cat((tens_l_rs, pred_ab), dim=1)
    real_img = torch.cat((tens_l_rs, tens_ab_rs), dim=1)
    # Discriminator predictions
    disc_fake = disc(fake_img)
    disc_real = disc(real_img)

    # Create real and fake labels
    real_labels = torch.ones(disc_real.size(), device=device)
    fake_labels = torch.zeros(disc_fake.size(), device=device)

    criterion_bce = torch.nn.BCELoss()

    # Compute discriminator loss
    loss_disc_real = criterion_bce(disc_real, real_labels)
    loss_disc_fake = criterion_bce(disc_fake, fake_labels)
    loss_disc = (loss_disc_real + loss_disc_fake) / 2

    # Backprop and optimize discriminator
    optimizer_disc.zero_grad()
    loss_disc.backward()
    optimizer_disc.step()

    return loss_disc


def train_gen(generator, disc, tens_l_rs, tens_ab_rs, optimizer_gen, device):
    pred_ab = generator(tens_l_rs)
    fake_img = torch.cat((tens_l_rs, pred_ab), dim=1)

    # Discriminator's output for fake images
    disc_fake = disc(fake_img)

    real_labels = torch.ones(disc_fake.size(), device=device)

    criterion_bce = torch.nn.BCELoss()
    criterion_mse = torch.nn.MSELoss()

    # Compute generator loss (adversarial + pixel loss)
    loss_gen_adv = criterion_bce(disc_fake, real_labels)
    loss_gen_pixel = criterion_mse(pred_ab, tens_ab_rs)
    loss_gen = loss_gen_adv + loss_gen_pixel

    # Backprop and optimize generator
    optimizer_gen.zero_grad()
    loss_gen.backward()
    optimizer_gen.step()

    return loss_gen


def train_colorization(
    train_dir,
    val_dir,
    epochs=10,
    batch_size=16,
    lr=0.0002,
    HW=(256, 256),
    use_gpu=True,
    checkpoint_path=None,
):
    device = torch.device(
        "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    )

    # Load data
    data_loader = get_data_loader(
        batch_size=batch_size, root_dir=train_dir, HW=HW
    )
    val_loader = get_data_loader(
        batch_size=batch_size, root_dir=val_dir, HW=HW
    )

    # Initialize models and optimizers
    generator = ECCVGenerator().to(device)
    disc = discriminator().to(device)

    # Initialize optimizers
    optimizer_gen = Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_disc = Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))

    criterion_mse = torch.nn.MSELoss()

    # Train models
    generator.train()
    disc.train()

    all_gen_losses = []
    all_disc_losses = []
    all_val_losses = []
    best_val_loss = float("inf")
    start_epoch = 0

    # Load from checkpoint if provided
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        generator.load_state_dict(checkpoint["generator_state_dict"])
        disc.load_state_dict(checkpoint["discriminator_state_dict"])
        optimizer_gen.load_state_dict(checkpoint["optimizer_gen_state_dict"])
        optimizer_disc.load_state_dict(checkpoint["optimizer_disc_state_dict"])
        all_gen_losses = checkpoint.get("all_gen_losses", [])
        all_disc_losses = checkpoint.get("all_disc_losses", [])
        all_val_losses = checkpoint.get("all_val_losses", [])
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        start_epoch = checkpoint["epoch"]
        print(f"Resumed training from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        epoch_loss_gen = 0
        epoch_loss_disc = 0

        tqdm_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in tqdm_bar:
            tens_l_rs = batch["L_resized"].to(device)
            tens_ab_rs = batch["AB_resized"].to(device)

            # ---------------------
            # Train Discriminator
            # ---------------------
            loss_disc = train_disc(
                generator=generator,
                disc=disc,
                tens_l_rs=tens_l_rs,
                tens_ab_rs=tens_ab_rs,
                optimizer_disc=optimizer_disc,
                device=device,
            )

            # ---------------------
            # Train Generator
            # ---------------------
            loss_gen = train_gen(
                generator=generator,
                disc=disc,
                tens_l_rs=tens_l_rs,
                tens_ab_rs=tens_ab_rs,
                optimizer_gen=optimizer_gen,
                device=device,
            )

            # Update epoch losses
            epoch_loss_gen += loss_gen.item()
            epoch_loss_disc += loss_disc.item()

            # Update progress bar
            tqdm_bar.set_postfix(
                {"Gen Loss": loss_gen.item(), "Disc Loss": loss_disc.item()}
            )

        # Average losses for the epoch
        avg_loss_gen = epoch_loss_gen / len(data_loader)
        avg_loss_disc = epoch_loss_disc / len(data_loader)
        all_gen_losses.append(avg_loss_gen)
        all_disc_losses.append(avg_loss_disc)

        print(
            f"Epoch {epoch+1}/{epochs}, Gen Loss: {avg_loss_gen:.4f}, Disc Loss: {avg_loss_disc:.4f}"
        )

        # ---------------------
        # Validation Phase
        # ---------------------
        generator.eval()  # Set generator to evaluation mode
        val_loss = 0
        with torch.no_grad():  # Disable gradient computation
            for val_batch in val_loader:
                val_l_rs = val_batch["L_resized"].to(device)
                val_ab_rs = val_batch["AB_resized"].to(device)
                val_pred_ab = generator(val_l_rs)
                val_loss += criterion_mse(val_pred_ab, val_ab_rs).item()

        val_loss /= len(val_loader)
        all_val_losses.append(val_loss)
        print(f"Validation Loss: {val_loss:.4f}")
        generator.train()  # Set generator back to train mode

        # Save image every 5 epoch
        if epoch % 5 == 0:
            visualize_hardcoded_image(
                generator=generator,
                image_path="imgs/ansel_adams.jpg",
                HW=(256, 256),  # Match your training data resolution
                epoch=epoch + 1,
                device=device,
                save_path="visualizations",  # Directory to save visualizations
            )

        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Define the path for the single checkpoint file
        checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")

        # Save the latest checkpoint (overwrites the previous one)
        torch.save(
            {
                "epoch": epoch + 1,
                "generator_state_dict": generator.state_dict(),
                "discriminator_state_dict": disc.state_dict(),
                "optimizer_gen_state_dict": optimizer_gen.state_dict(),
                "optimizer_disc_state_dict": optimizer_disc.state_dict(),
                "loss_gen": epoch_loss_gen / len(data_loader),
                "loss_disc": epoch_loss_disc / len(data_loader),
                "all_gen_losses": all_gen_losses,
                "all_disc_losses": all_disc_losses,
                "all_val_losses": all_val_losses,
                "best_val_loss": best_val_loss,
            },
            checkpoint_path,
        )

        print(f"Checkpoint updated at {checkpoint_path}")

        # ---------------------
        # Plot Training and Validation Losses
        # ---------------------

        # Ensure the visualizations directory and plots subdirectory exist
        visualizations_dir = "visualizations"
        plots_dir = os.path.join(visualizations_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Plot Generator Loss
        plt.figure(figsize=(10, 6))
        plt.plot(all_gen_losses, label="Generator Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Generator Loss")
        plt.grid()
        gen_loss_plot_path = os.path.join(plots_dir, "gen_loss_plot.png")
        plt.savefig(gen_loss_plot_path)
        plt.close()

        # Plot Discriminator Loss
        plt.figure(figsize=(10, 6))
        plt.plot(all_disc_losses, label="Discriminator Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Discriminator Loss")
        plt.grid()
        disc_loss_plot_path = os.path.join(plots_dir, "disc_loss_plot.png")
        plt.savefig(disc_loss_plot_path)
        plt.close()

        # Plot Validation Loss
        plt.figure(figsize=(10, 6))
        plt.plot(all_val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Validation Loss")
        plt.grid()
        val_loss_plot_path = os.path.join(plots_dir, "val_loss_plot.png")
        plt.savefig(val_loss_plot_path)
        plt.close()

        # Plot Combined Losses
        plt.figure(figsize=(10, 6))
        plt.plot(all_gen_losses, label="Generator Loss")
        plt.plot(all_disc_losses, label="Discriminator Loss")
        plt.plot(all_val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Losses")
        plt.legend()
        plt.grid()
        combined_loss_plot_path = os.path.join(
            plots_dir, "combined_loss_plot.png"
        )
        plt.savefig(combined_loss_plot_path)
        plt.close()
        print("Updated plots")
