# import torch
# from torch.optim import Adam
# from tqdm import tqdm
# from .data_loader import get_data_loader
# from .eccv16 import ECCVGenerator

# def train_colorization(root_dir, epochs=10, batch_size=16, lr=0.0002, HW=(256, 256), use_gpu=True):
#     device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
#     data_loader = get_data_loader(batch_size=batch_size, root_dir=root_dir, HW=HW)
#     generator = ECCVGenerator().to(device)
#     optimizer = Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
#     criterion = torch.nn.MSELoss()

#     generator.train()
#     for epoch in range(epochs):
#         epoch_loss = 0
#         tqdm_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}")
#         for batch in tqdm_bar:
#             tens_l_rs = batch["L_resized"].to(device)
#             tens_ab_rs = batch["AB_resized"].to(device)
#             # print(tens_l_rs.shape)
#             optimizer.zero_grad()
#             pred_ab = generator(tens_l_rs)
#             loss = criterion(pred_ab, tens_ab_rs)
#             epoch_loss += loss.item()
#             loss.backward()
#             optimizer.step()

#             tqdm_bar.set_postfix({"Loss": loss.item()})
#         print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(data_loader):.4f}")

#     torch.save(generator.state_dict(), "colorization_generator.pth")

import torch
from torch.optim import Adam
from tqdm import tqdm
from .data_loader import get_data_loader
from .eccv16 import ECCVGenerator
from .discriminator import discriminator


def train_colorization(root_dir, epochs=10, batch_size=16, lr=0.0002, HW=(256, 256), use_gpu=True):
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    # Load data
    data_loader = get_data_loader(batch_size=batch_size, root_dir=root_dir, HW=HW)

    # Initialize models
    generator = ECCVGenerator().to(device)
    disc = discriminator().to(device)

    # Initialize optimizers
    optimizer_gen = Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_disc = Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))

    # Define loss functions
    criterion_mse = torch.nn.MSELoss()  # For generator's pixel loss
    criterion_bce = torch.nn.BCELoss()  # For discriminator's adversarial loss

    # Train models
    generator.train()
    disc.train()

    for epoch in range(epochs):
        epoch_loss_gen = 0
        epoch_loss_disc = 0

        tqdm_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in tqdm_bar:
            tens_l_rs = batch["L_resized"].to(device)  # Input grayscale image
            tens_ab_rs = batch["AB_resized"].to(device)  # Ground truth color channels

            # ---------------------
            # Train Discriminator
            # ---------------------
            # Generate fake colorizations
            pred_ab = generator(tens_l_rs).detach()  # Detach to avoid gradients flowing to generator
            # fake_img = torch.cat((tens_l_rs, pred_ab), dim=1)  # Fake input for discriminator
            # real_img = torch.cat((tens_l_rs, tens_ab_rs), dim=1)  # Real input for discriminator
            # Concatenate L and AB channels to form 6-channel input
            fake_img = torch.cat((tens_l_rs, pred_ab, pred_ab), dim=1)  # Shape: [batch_size, 5, height, width]
            real_img = torch.cat((tens_l_rs, tens_ab_rs, tens_ab_rs), dim=1)  # Shape: [batch_size, 5, height, width]
            # Discriminator predictions
            disc_fake = disc(fake_img)
            disc_real = disc(real_img)

            # Create real and fake labels
            real_labels = torch.ones(disc_real.size(), device=device)  # Real labels = 1
            fake_labels = torch.zeros(disc_fake.size(), device=device)  # Fake labels = 0

            # Compute discriminator loss
            loss_disc_real = criterion_bce(disc_real, real_labels)
            loss_disc_fake = criterion_bce(disc_fake, fake_labels)
            loss_disc = (loss_disc_real + loss_disc_fake) / 2

            # Backprop and optimize discriminator
            optimizer_disc.zero_grad()
            loss_disc.backward()
            optimizer_disc.step()

            # ---------------------
            # Train Generator
            # ---------------------
            # Generate fake colorizations
            pred_ab = generator(tens_l_rs)
            fake_img = torch.cat((tens_l_rs, pred_ab, pred_ab), dim=1)  # Shape: [batch_size, 5, height, width]

            # Discriminator's output for fake images
            disc_fake = disc(fake_img)

            # Compute generator loss (adversarial + pixel loss)
            loss_gen_adv = criterion_bce(disc_fake, real_labels)  # Adversarial loss
            loss_gen_pixel = criterion_mse(pred_ab, tens_ab_rs)  # Pixel-wise loss
            loss_gen = loss_gen_adv + loss_gen_pixel

            # Backprop and optimize generator
            optimizer_gen.zero_grad()
            loss_gen.backward()
            optimizer_gen.step()

            # Update epoch losses
            epoch_loss_gen += loss_gen.item()
            epoch_loss_disc += loss_disc.item()

            # Update progress bar
            tqdm_bar.set_postfix({"Gen Loss": loss_gen.item(), "Disc Loss": loss_disc.item()})

        print(f"Epoch {epoch+1}/{epochs}, Gen Loss: {epoch_loss_gen / len(data_loader):.4f}, Disc Loss: {epoch_loss_disc / len(data_loader):.4f}")

    # Save generator model
    torch.save(generator.state_dict(), "colorization_generator.pth")
    # Save discriminator model (optional)
    torch.save(disc.state_dict(), "colorization_discriminator.pth")

