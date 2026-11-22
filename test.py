import torch
from torch import nn
from unet import Unet as unet
from data_loader import *
from torch import optim
from collections import OrderedDict
from torch.optim.lr_scheduler import StepLR
from discriminator import *

def load_checkpoint(path, model, optimizer):
    """ loading a checkpoint """
    checkpoint = torch.load(path, weights_only=False, map_location=torch.device('cpu'))
    state_dict = checkpoint["model_state_dict"]

    # FROM CHATGBT TO JUST GETTING IT DONE
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")  # strip only at the start
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch     = checkpoint.get("epoch", 0)
    loss      = checkpoint.get("loss", None)
    val_loss  = checkpoint.get("val_loss", None)
    run_id    = checkpoint.get("run_id", None)

    return model, optimizer, epoch, loss, val_loss, run_id

def load_checkpoint2(path, model, optimizer):
    """ loading a checkpoint """
    checkpoint = torch.load(path, weights_only=False, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    #scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    val_loss = checkpoint["val_loss"]
    run_id = checkpoint["run_id"]

    return model, optimizer, epoch, loss, val_loss, run_id

def load_checkpoint3(path, model):
    """ loading a checkpoint """
    checkpoint = torch.load(path, weights_only=False, map_location=torch.device('cpu'))

    def strip_module(sd):
        # if keys start with "module.", remove that prefix
        from collections import OrderedDict
        new_sd = OrderedDict()
        for k, v in sd.items():
            new_k = k.replace("module.", "")
            new_sd[new_k] = v
        return new_sd

    # --- load generator ---
    gen_sd = checkpoint["gen_state_dict"]
    try:
        model.load_state_dict(gen_sd)
    except RuntimeError:
        # probably saved with DataParallel → strip "module."
        model.load_state_dict(strip_module(gen_sd))

    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    val_loss = checkpoint["val_loss"]
    run_id = checkpoint["run_id"]

    return model, epoch, loss, val_loss, run_id

def test(model, device, val_loader):
    """ validation loop """
    model.eval()
    lossfunction = nn.L1Loss()
    total_loss = 0
    total_batch_size = 0
    #psnr_metric = PSNR(data_range=1.0)

    with torch.no_grad():
        loss = 0
        for batch, (x,y) in enumerate(val_loader):
            if batch % 2000 == 0:
                print(f"Progress: {batch}/300000")
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            batch_loss = lossfunction(y, output).item()
            #psnr_metric.update((output, y))
            batch_size = x.size(0)

            total_loss += batch_loss * batch_size
            total_batch_size += batch_size

        #psnr_value = psnr_metric.compute()
        #print("PSNR:", psnr_value)

        loss = total_loss / total_batch_size
        print("MAE: ", loss, "\n")
    return loss

def save_test_image(pred_tensor, target_tensor, i):
    pred_tensor = torch.clamp(pred_tensor, 0.0, 1.0)
    pred_img = transforms.functional.to_pil_image(pred_tensor[0].detach().cpu())
    pred_img.save(f"test_output_images/places_cropped/pretrained_test_pred_img{i}.jpg")
    target_tensor = torch.clamp(target_tensor, 0.0, 1.0)
    target_img = transforms.functional.to_pil_image(target_tensor[0].detach().cpu())
    target_img.save(f"test_output_images/places_cropped/pretrained_test_x_img{i}.jpg")

def main():
    # import model
    torch.manual_seed(42)
    model = unet()
    
    #model = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=False, scale=1)
    #model.inc.double_conv[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #model.outc.conv = nn.Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    data_path = "data"
    print("Klar")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("cuda")
    elif torch.mps.is_available():
        device = torch.device("mps")
        print("mps")
    else:
        device = torch.device("cpu")
        print("cpu")

    model = model.to(device)
    #optimizer = optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)


    #checkpoint_path = "unet_places_cropped_pretrained.tar"
    #checkpoint_path = "unet_places_cropped.tar"
    checkpoint_path = "checkpoint/patch/patch_unet_places_cropped_adam_step.tar"
    model, epoch, loss_check, val_loss, run_id = load_checkpoint3(checkpoint_path, model)
    print("checkpoint loaded")
    print(f"Checkpoint after {epoch}, with validation loss {val_loss}")
    
    test_data = gray_color_data(data_path, split="test", train=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
    print("data loaded")
    loss = test(model, device, test_loader)
    print(loss)

    for i, (x, y) in enumerate(test_loader):
                test_img = x.to(device)
                model.eval()
                test_pred = model(test_img)
                save_test_image(test_pred, y, i)
                if i > 10:
                    break

if __name__ == "__main__":
    main()