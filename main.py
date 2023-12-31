import argparse
from pathlib import Path
import sys
import os
import torchaudio

from compress import compress, decompress, MODELS
from utils import save_audio, convert_audio
# Test the model for 1 audio
# Compression and decompression scripts
SUFFIX = 'ecdc'

def get_parser():
    parser = argparse.ArgumentParser(
        'encodec',
        description='High fidelity neural audio codec. '
                    'If input is a .ecdc, decompresses it. '
                    'If input is .wav, compresses it. If output is also wav, '
                    'do a compression/decompression cycle.')
    parser.add_argument(
        'input', type=Path,
        help='Input file, whatever is supported by torchaudio on your system.')
    parser.add_argument(
        'output', type=Path, nargs='?',
        help='Output file, otherwise inferred from input file.')
    parser.add_argument(
        '-b', '--bandwidth', type=float, default=6, choices=[1.5, 3., 6., 12., 24.],
        help='Target bandwidth (1.5, 3, 6, 12 or 24). 1.5 is not supported with --hq.')
    parser.add_argument(
        '-q', '--hq', action='store_true',
        help='Use HQ stereo model operating on 48 kHz sampled audio.')
    parser.add_argument(
        '-l', '--lm', action='store_true',
        help='Use a language model to reduce the model size (5x slower though).')
    parser.add_argument(
        '-f', '--force', action='store_true',
        help='Overwrite output file if it exists.')
    parser.add_argument(
        '-s', '--decompress_suffix', type=str, default='_decompressed',
        help='Suffix for the decompressed output file (if no output path specified)')
    parser.add_argument(
        '-r', '--rescale', action='store_true',
        help='Automatically rescale the output to avoid clipping.')
    parser.add_argument(
        '-m','--model_name', type=str, default='encodec_24khz',
        help='support encodec_24khz,encodec_48khz,my_encodec')
    parser.add_argument(
        '-c','--checkpoint', type=str, 
        help='if use my_encodec, please input checkpoint')
    return parser


def fatal(*args):
    print(*args, file=sys.stderr)
    sys.exit(1)


def check_output_exists(args):
    if not Path(args.output).parent.exists():
        fatal(f"Output folder for {args.output} does not exist.")
    if Path(args.output).exists() and not args.force:
        fatal(f"Output file {args.output} exist. Use -f / --force to overwrite.")


def check_clipping(wav, args):
    if args.rescale:
        return
    mx = wav.abs().max()
    limit = 0.99
    if mx > limit:
        print(
            f"Clipping!! max scale {mx}, limit is {limit}. "
            "To avoid clipping, use the `-r` option to rescale the output.",
            file=sys.stderr)

def suffix(file: str):
    return file.split('.')[-1]

def main(args,model):

    if args.return_object:
        print("Return object True")
        if suffix(args.input).lower() == SUFFIX:
            # Decompress
            out, out_sample_rate = decompress(model, args.input.read_bytes())
            return out, out_sample_rate
        if suffix(args.input).lower() in ['wav', 'flac']:
            # Compress + Decompress
            wav, sr = torchaudio.load(args.input)
            wav = convert_audio(wav, sr, model.sample_rate, model.channels)
            compressed = compress(model, wav, use_lm=args.lm)
            out, out_sample_rate = decompress(model, compressed)
            return out, out_sample_rate
        else:
            fatal(f"Input extension must be one of {[SUFFIX, 'wav', 'flac']}")

    # if args.input.suffix.lower() == SUFFIX:
    if suffix(args.input).lower() == SUFFIX:
        # Decompression
        if args.output is None:
            args.output = args.input.with_name(args.input.stem + args.decompress_suffix).with_suffix('.wav')
        elif suffix(args.output).lower() != 'wav':
            fatal("Output extension must be wav")
        check_output_exists(args)
        out, out_sample_rate = decompress(model, args.input.read_bytes())
        check_clipping(out, args)
        save_audio(out, args.output, out_sample_rate, rescale=args.rescale)
    else:
        # Compression
        if args.output is None:
            args.output = args.input.with_suffix(SUFFIX)
        elif suffix(args.output).lower() not in [SUFFIX, 'wav', 'flac']:
            fatal(f"Output extension must be one of {[SUFFIX, 'wav', 'flac']}")
        check_output_exists(args)

        wav, sr = torchaudio.load(args.input)
        wav = convert_audio(wav, sr, model.sample_rate, model.channels)
        compressed = compress(model, wav, use_lm=args.lm)
        if suffix(args.output).lower() == SUFFIX:
            args.output.write_bytes(compressed)
        else:
            # Directly run decompression stage
            assert suffix(args.output).lower() == 'wav'
            out, out_sample_rate = decompress(model,compressed)
            check_clipping(out, args)
            save_audio(out, args.output, out_sample_rate, rescale=args.rescale)

def cli_main(args):
    if args.hq:
        model_name = 'encodec_48khz'
    else:
        model_name = args.model_name

    if model_name == 'my_encodec':
        model = MODELS[model_name](args.checkpoint)
    elif model_name == 'encodec_bw':
        model = MODELS[model_name](args.checkpoint,[args.bandwidth])
    else:
        model = MODELS[model_name]()
    
    print(f"-------------USE {model_name} MODEL-------------")

    if args.bandwidth not in model.target_bandwidths:
        fatal(f"Bandwidth {args.bandwidth} is not supported by the model {model_name}")
    model.set_target_bandwidth(args.bandwidth)
    
    if Path(args.input).is_dir():
        output_root = args.output
        input_root = args.input
        if not Path(output_root).exists():
            Path(output_root).mkdir(parents=True)
        for root, dirs, files in os.walk(input_root):
            for file in files:
                if file.lower().endswith(('.flac', '.wav')):
                    print(f"Processing {file}")
                    args.input = os.path.join(root, file)
                    output_name = file.split('.')[0] + "-" + str(args.bandwidth) + ".wav"
                    output_wav_file = os.path.join(output_root, output_name)
                    args.output = output_wav_file
                    main(args, model)
    elif Path(args.input).is_file():
        # If processing one file, and we need to pass the objects (decoded eventually) back
        if args.return_object:
            return main(args, model)
        else:
            main(args, model)

if __name__ == '__main__':
    args = get_parser().parse_args()
    if not Path(args.input).exists():
        fatal(f"Input file {args.input} does not exist.")
    cli_main(args)
