import torch
from audio_to_mel import Audio2Mel

def total_loss(fmap_real, logits_fake, fmap_fake, input_wav, output_wav, sample_rate=24000):
    """This function is used to compute the total loss of the encodec generator.
        Loss = lambda_t * L_t + lambda_f * L_f + lambda_g * L_g + lambda_feat * L_feat
        L_t: time domain loss | L_f: frequency domain loss | L_g: generator loss | L_feat: feature loss
        lambda_t = 0.1        | lambda_f = 1               | lambda_g = 3        | lambda_feat = 3
    Args:
        fmap_real (list): fmap_real is the output of the discriminator when the input is the real audio. 
            len(fmap_real) = len(fmap_fake) = disc.num_discriminators = 3
        logits_fake (_type_): logits_fake is the list of every sub discriminator output of the Multi discriminator 
            logits_fake, _ = disc_model(model(input_wav)[0].detach())
        fmap_fake (_type_): fmap_fake is the output of the discriminator when the input is the fake audio.
            fmap_fake = disc_model(model(input_wav)[0]) = disc_model(reconstructed_audio)
        input_wav (tensor): input_wav is the input audio of the generator (GT audio)
        output_wav (tensor): output_wav is the output of the generator (output = model(input_wav)[0])
        sample_rate (int, optional): Defaults to 24000.

    Returns:
        loss: total loss
    """
    relu = torch.nn.ReLU()
    l1Loss = torch.nn.L1Loss(reduction='mean')
    l2Loss = torch.nn.MSELoss(reduction='mean')
    device = "cuda" if torch.cuda.is_available() else "cpu"

    l_f = torch.tensor([0.0], device=device, requires_grad=True)
    l_g = torch.tensor([0.0], device=device, requires_grad=True)
    l_feat = torch.tensor([0.0], device=device, requires_grad=True)

    # Time domain loss, output_wav is the output of the generator
    l_t = l1Loss(input_wav, output_wav)

    # Frequency domain loss, window length is 2^i, hop length is 2^i/4. combine l1 and l2 loss
    # e=5,...,11
    for i in range(5, 12):
        fft = Audio2Mel(n_fft=2 ** i, win_length=2 ** i, hop_length=(2 ** i) // 4,
                        n_mel_channels=64, sampling_rate=sample_rate)
        l_f = l_f + l1Loss(fft(input_wav), fft(output_wav)) + l2Loss(fft(input_wav), fft(output_wav))

    # Generator loss and feat loss, k ~ discriminator, l ~ layer
    # Basic setup: K = 3, L = 5
    # KL = len(fmap_real[0]) * len(fmap_real) = 3 * 5
    # D_k(hat_x) = logits_fake[k],
    # D_k^l(x) = fmap_real[k][l], D_k^l(hat_x) = fmap_fake[k][l]
    # l_g = sum: max(0, 1 - D_k(hat_x)) / K,
    # l_feat = sum: |D_k^l(x) - D_k^l(hat_x)| / |D_k^l(x)| / KL

    # For disc
    for disc in range(len(fmap_real)):
        l_g = l_g + torch.mean(relu(1 - logits_fake[disc])) / len(logits_fake)
        # For layer in disc
        for layer in range(len(fmap_real[disc])):
            # l_feat = l_feat + l1Loss(fmap_real[disc][layer].detach(),
            # fmap_fake[disc][layer]) / torch.mean(torch.abs(fmap_real[disc][layer].detach()))
            l_feat = l_feat + \
                     l1Loss(fmap_real[disc][layer], fmap_fake[disc][layer]) / \
                     torch.mean(torch.abs(fmap_real[disc][layer]))

    K = len(fmap_real)
    L = len(fmap_real) * len(fmap_real[0])

    # Adjust losses with the weights
    loss = 3*l_g/K + 3*l_feat/L + 0.1*l_t + 1*l_f

    return loss

def disc_loss(logits_real, logits_fake):
    """This function is used to compute the loss of the discriminator.
        l_d = sum: max(0, 1 - D_k(x)) + max(0, 1 + D_k(hat_x)) / K,
        K = disc.num_discriminators = len(logits_real) = len(logits_fake) = 3
    Args:
        logits_real (List[torch.Tensor]): logits_real = disc_model(input_wav)[0]
        logits_fake (List[torch.Tensor]): logits_fake = disc_model(model(input_wav)[0])[0]
    
    Returns:
        loss_multi_disc: discriminator loss
    """
    relu = torch.nn.ReLU()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_multi_disc = torch.tensor([0.0], device=device, requires_grad=True)
    for disc in range(len(logits_real)):
        # instead of max function we could use RELU here, which for pos values outputs positives
        # And for negative stays zero
        loss_disc = torch.mean(relu(1-logits_real[disc])) + torch.mean(relu(1+logits_fake[disc]))
        loss_multi_disc = loss_multi_disc + loss_disc
    loss_multi_disc = loss_multi_disc / len(logits_real)
    return loss_multi_disc
