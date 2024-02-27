import os
import torch
import torch.optim as optim
import customAudioDataset as data
from customAudioDataset import collate_fn
from datasets.generate_desc_file import generate_csv, split_train_test_csv, duplicate_paths_in_csv
from utils import set_seed, save_master_checkpoint, start_dist_train, count_parameters
from msstftd import MultiScaleSTFTDiscriminator
from losses import disc_loss
from scheduler import WarmupCosineLrScheduler
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
import hydra
import logging
import warnings
import shutil
import tarfile
import zipfile

warnings.filterwarnings("ignore")

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def compress_and_move(input_dirs, output_dir, cut):
    """
    cut: int, denotes number of audios to take from each directory
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over each input directory
    for input_dir in input_dirs:
        # Get the list of audio files (FLAC and WAV) in the input directory
        audio_files = [file for file in os.listdir(input_dir) if file.endswith(('.flac', '.wav'))][:cut]
        assert len(audio_files) == cut
        # Create a zip archive for each input directory
        archive_name = os.path.basename(input_dir) + '.zip'
        archive_path = os.path.join(input_dir, archive_name)
        with zipfile.ZipFile(archive_path, 'w') as zipf:
            for audio_file in audio_files:
                audio_path = os.path.join(input_dir, audio_file)
                zipf.write(audio_path, arcname=audio_file)

        # Move the zip archive to the output directory
        shutil.move(archive_path, output_dir)

    # Extract each archive in the output directory
    for archive_file in os.listdir(output_dir):
        if archive_file.endswith('.zip'):
            archive_path = os.path.join(output_dir, archive_file)
            with zipfile.ZipFile(archive_path, 'r') as zipf:
                zipf.extractall(output_dir)
            os.remove(archive_path)  # Remove the zip archive after extraction


def train_one_epoch(epoch, optimizer_disc, disc_model,
                    trainloader_real, trainloader_fake, config, disc_scheduler,
                    scaler=None, scaler_disc=None, writer=None):
    """Train one epoch function
    Args:
        epoch (int): current epoch
        optimizer_disc (_type_): discriminator optimizer
        disc_model (_type_): discriminator model
        trainloader_real (_type_): real dataloader
        trainloader_fake (_type_): fake dataloader
        config (_type_): hydra config file
        disc_scheduler (_type_): adjust discriminator model learning rate
        warmup_scheduler (_type_): warmup learning rate
    """
    disc_model.train()
    data_length = len(trainloader_real)

    # Initialize variables to accumulate losses
    accumulated_loss_disc = 0.0

    for fake_data, real_data in zip(trainloader_fake, trainloader_real):

        idx_fake, fake_inputs = fake_data

        idx_real, real_inputs = real_data

        # Input: [Batch, Channels, Time]
        if torch.cuda.is_available():
            fake_inputs = fake_inputs.contiguous().cuda()
            real_inputs = real_inputs.contiguous().cuda()
        else:
            fake_inputs = fake_inputs.contiguous()
            real_inputs = real_inputs.contiguous()

        # Train discriminator (after warmup)
        optimizer_disc.zero_grad()
        logits_real, _ = disc_model(real_inputs)
        logits_fake, _ = disc_model(fake_inputs)
        loss_disc = disc_loss(logits_real, logits_fake)
        loss_disc.backward()
        optimizer_disc.step()
        # Accumulate discriminator loss
        accumulated_loss_disc += loss_disc.item()

        disc_scheduler.step()

        log_msg = f"loss_disc: {accumulated_loss_disc / (idx_real + 1) :.4f}"
        writer.add_scalar('Train/Loss_Disc', accumulated_loss_disc / (idx_real + 1),
                          (epoch - 1) * len(trainloader_real) + idx_real)
        logger.info(log_msg)


@torch.no_grad()
def test(epoch, disc_model, testloader_real, testloader_fake, config, writer):
    disc_model.eval()
    for fake_data, real_data in zip(testloader_fake, testloader_real):

        idx_fake, fake_inputs = fake_data

        idx_real, real_inputs = real_data

        # [B, 1, T]: eg. [2, 1, 203760]
        if torch.cuda.is_available():
            fake_inputs = fake_inputs.contiguous().cuda()
            real_inputs = real_inputs.contiguous().cuda()
        else:
            fake_inputs = fake_inputs.contiguous()
            real_inputs = real_inputs.contiguous()

        logits_real, _ = disc_model(real_inputs)
        logits_fake, _ = disc_model(fake_inputs)

        # compute discriminator loss
        loss_disc = disc_loss(logits_real, logits_fake)

    if not config.distributed.data_parallel or dist.get_rank() == 0:
        log_msg = (f'| TEST | epoch: {epoch} | loss_disc: {loss_disc.item():.4f}')
        writer.add_scalar('Test/Loss_Disc', loss_disc.item(), epoch)
        logger.info(log_msg)


def train(local_rank, world_size, config, tmp_file=None):
    """train main function
    args:
        world_size - how many GPUs
        local_rank - rank of the current process (should be between 1 and world_size), for parallel
    """
    checkpoint_folder = os.path.normpath(config.checkpoint.save_folder)
    # set logger
    file_handler = logging.FileHandler(
        f"{checkpoint_folder}/train_encodec_bs{config.datasets.batch_size}_lr{config.optimization.disc_lr}.log")
    formatter = logging.Formatter('%(asctime)s: %(levelname)s: [%(filename)s: %(lineno)d]: %(message)s')
    file_handler.setFormatter(formatter)

    # print to screen
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    # add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # set seed
    if config.common.seed is not None:
        set_seed(config.common.seed)

    # send thed data to the node
    WORK = "/home/woody/iwi1/iwi1010h"
    OUTPUT_DIR = WORK + "/Compression_analysis"
    OUTPUT_DIR_SOUNDSTREAM = OUTPUT_DIR + "/SoundStream"
    OUTPUT_DIR_ENCODEC = OUTPUT_DIR + "/Encodec"
    OUTPUT_DIR_NEW_ENCODEC = OUTPUT_DIR_ENCODEC + "/New_encodec"
    TMPDIR = os.environ.get('TMPDIR')
    SLURM_JOBID = os.environ.get('SLURM_JOBID')

    # Move Fake
    input_directories = [
        os.path.normpath(OUTPUT_DIR_SOUNDSTREAM),
        os.path.normpath(OUTPUT_DIR_ENCODEC+"/24.0"),
        os.path.normpath(OUTPUT_DIR_NEW_ENCODEC)
    ]
    output_directory = os.path.normpath(os.path.join(TMPDIR,SLURM_JOBID, "Fake_dir"))
    compress_and_move(input_directories, output_directory, cut=500) # 500 audios each
    print("Moved fake")

    # Create description csv files FAKE
    input_dir_for_csv = output_directory
    csv_file = os.path.normpath(os.path.join(TMPDIR, SLURM_JOBID, "disc_fake.csv"))
    generate_csv(input_dir_for_csv, csv_file)
    fake_train_csv, fake_test_csv = split_train_test_csv(csv_file)
    config.datasets.disc_train_real_csv = fake_train_csv
    config.datasets.disc_test_real_csv = fake_test_csv
    print("Generated Real csv files")

    # Move Real
    DATA_IN_ONE_DIR = OUTPUT_DIR + "/Data"
    input_directories = [
        os.path.normpath(DATA_IN_ONE_DIR)
    ]
    output_directory = os.path.normpath(os.path.join(TMPDIR, SLURM_JOBID, "Real_dir"))
    compress_and_move(input_directories, output_directory, cut=1500)# 1500 real audios
    print("Moved real")

    # Create description csv files REAL
    input_dir_for_csv = output_directory
    csv_file = os.path.normpath(os.path.join(TMPDIR, SLURM_JOBID, "disc_real.csv"))
    generate_csv(input_dir_for_csv, csv_file)
    # Double the size of the real
    # duplicate_paths_in_csv(csv_file)
    real_train_csv, real_test_csv = split_train_test_csv(csv_file)
    config.datasets.disc_train_real_csv = real_train_csv
    config.datasets.disc_test_real_csv = real_test_csv
    print("Generated Real csv files")

    # set train and test datasets for real and fake
    trainset_real = data.CustomAudioDataset(config=config, mode="disc_real")
    trainset_fake = data.CustomAudioDataset(config=config, mode="disc_fake")
    testset_real = data.CustomAudioDataset(config=config, mode='disc_real_test')
    testset_fake = data.CustomAudioDataset(config=config, mode='disc_fake_test')

    disc_model = MultiScaleSTFTDiscriminator(filters=config.model.filters,
                                             hop_lengths=config.model.disc_hop_lengths,
                                             win_lengths=config.model.disc_win_lengths,
                                             n_ffts=config.model.disc_n_ffts)

    # log model, disc model parameters and train mode
    logger.info(config)
    logger.info(
        f"Disc Model Parameters: {count_parameters(disc_model)}")

    # If continue training
    resume_epoch = 0
    if config.checkpoint.resume:
        # check the checkpoint_path
        assert config.checkpoint.checkpoint_path != '', "resume path is empty"
        assert config.checkpoint.disc_checkpoint_path != '', "disc resume path is empty"

        logger.info(f"Resuming training!")
        # Why map_location CPU Info: You can call torch.load(.., map_location='cpu') and then load_state_dict()
        # to avoid GPU RAM surge when loading a model checkpoint.
        disc_model_checkpoint = torch.load(config.checkpoint.disc_checkpoint_path, map_location='cpu')
        disc_model.load_state_dict(disc_model_checkpoint['model_state_dict'])
        resume_epoch = disc_model_checkpoint['epoch']
        if resume_epoch >= config.common.max_epoch:
            raise ValueError(f"resume epoch {resume_epoch} is larger than total epochs {config.common.epochs}")
        logger.info(f"load checkpoints of disc_model, resume from {resume_epoch}")

    train_sampler = None
    test_sampler = None

    # Move to GPU
    if torch.cuda.is_available():
        disc_model.cuda()

    # Set up DataLoader
    trainloader_real = torch.utils.data.DataLoader(
        trainset_real,
        batch_size=config.datasets.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None), collate_fn=collate_fn,
        pin_memory=config.datasets.pin_memory)

    trainloader_fake = torch.utils.data.DataLoader(
        trainset_fake,
        batch_size=config.datasets.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None), collate_fn=collate_fn,
        pin_memory=config.datasets.pin_memory)

    testloader_real = torch.utils.data.DataLoader(
        testset_real,
        batch_size=config.datasets.batch_size,
        sampler=test_sampler,
        shuffle=False, collate_fn=collate_fn,
        pin_memory=config.datasets.pin_memory)

    testloader_fake = torch.utils.data.DataLoader(
        testset_fake,
        batch_size=config.datasets.batch_size,
        sampler=test_sampler,
        shuffle=False, collate_fn=collate_fn,
        pin_memory=config.datasets.pin_memory)

    logger.info(f"There are {len(trainloader_real)} batches in train real ")
    logger.info(f"There are {len(trainloader_fake)} batches in train fake ")
    logger.info(f"There are {len(testloader_real)} batches in test real")
    logger.info(f"There are {len(testloader_fake)} batches in test fake")

    # Set optimizer
    disc_params = [p for p in disc_model.parameters() if p.requires_grad]
    optimizer_disc = optim.Adam([{'params': disc_params, 'lr': config.optimization.disc_lr}], betas=(0.5, 0.9))

    disc_scheduler = WarmupCosineLrScheduler(optimizer_disc,
                                             max_iter=config.common.max_epoch * len(trainloader_real),
                                             eta_ratio=0.1,
                                             warmup_iter=config.lr_scheduler.warmup_epoch * len(trainloader_real),
                                             warmup_ratio=1e-4)

    # Scaler: (AutoMixPrecision) changing data types to speed up computation
    # Default: False
    scaler = GradScaler() if config.common.amp else None
    scaler_disc = GradScaler() if config.common.amp else None

    # If continue training: load optimizer and scheduler states from checkpoints
    if config.checkpoint.resume and 'scheduler_state_dict' in disc_model_checkpoint.keys():
        optimizer_disc.load_state_dict(disc_model_checkpoint['optimizer_state_dict'])
        disc_scheduler.load_state_dict(disc_model_checkpoint['scheduler_state_dict'])
        logger.info(f"load optimizer and disc_optimizer state_dict from {resume_epoch}")

    if not config.distributed.data_parallel or dist.get_rank() == 0:
        # Set up writer to log events
        writer = SummaryWriter(log_dir=f'{checkpoint_folder}/runs')
    else:
        writer = None

    # Start epoch is 1 if not resume
    start_epoch = max(1, resume_epoch + 1)

    for epoch in range(start_epoch, config.common.max_epoch + 1):
        train_one_epoch(
            epoch, optimizer_disc, disc_model,
            trainloader_real, trainloader_fake, config, disc_scheduler,
            scaler, scaler_disc, writer)
        if epoch % config.common.test_interval == 0:
            test(epoch, disc_model, testloader_real, testloader_fake, config, writer)
        # Save checkpoint and epoch
        if epoch % config.common.save_interval == 0:
            disc_model_to_save = disc_model.module if config.distributed.data_parallel else disc_model
            if not config.distributed.data_parallel or dist.get_rank() == 0:
                save_master_checkpoint(epoch, disc_model_to_save, optimizer_disc, disc_scheduler,
                                       f'{config.checkpoint.save_location}ep{epoch}_disc_lr{config.optimization.disc_lr}.pt')


# Since hydra is set, cur work dir is changed to the ones with the logs
@hydra.main(config_path='config', config_name='config')
def main(config):
    # Set the checkpoint folder
    checkpoint_folder = os.path.normpath(config.checkpoint.save_folder)
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    # Turn off cuda if not available
    if not torch.cuda.is_available():
        torch.backends.cudnn.enabled = False
        # set single gpu train
    train(1, 1, config)

if __name__ == '__main__':
    main()
