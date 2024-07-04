import torch
from snac import SNAC
import torchaudio
import json
from collections import deque

class SpeechTokenizer:
    def __init__(self, device='cpu') -> None:
        self.model = torch.compile(SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device))
        self.sample_rate = 24000
        self.device = device

    def encode(self, audio):
        with torch.inference_mode():
            codes = self.model.encode(audio)

        # Print the shape, length, and dimensions of the immediate tensors
        for i, code in enumerate(codes):
            print(f"Tensor {i} - Shape: {code.shape}, Length: {code.numel()}, Dimensions: {code.dim()}")

        with torch.no_grad():
            if 'cuda' in self.device:
                torch.cuda.empty_cache()
        return codes

    def decode(self, codes):
        codes = [c.clone().detach().to(self.device) for c in codes]

        # Print the shape, length, and dimensions of the immediate tensors
        for i, code in enumerate(codes):
            print(f"Tensor {i} - Shape: {code.shape}, Length: {code.numel()}, Dimensions: {code.dim()}")

        with torch.inference_mode():
            audio_hat = self.model.decode(codes)
        with torch.no_grad():
            if 'cuda' in self.device:
                torch.cuda.empty_cache()
        return audio_hat

    def save_to_json(self, codes, path):
        codes_list = [c.tolist() for c in codes]
        with open(path, 'w') as f:
            json.dump(codes_list, f)
        print(f"Encoded audio saved to {path}")

    def load_from_json(self, path):
        with open(path, 'r') as f:
            codes_list = json.load(f)
        codes = [torch.tensor(c) for c in codes_list]
        return codes

    # def flatten_tensors(self, tensors, separator=4097):
    #     """Simplified flattening of tensors into a flat list of integers."""
    #     flattened = [separator] + [item for tensor in tensors for sublist in tensor.tolist() for item in sublist]
    #     flattened.append(separator)
    #     return flattened
    def flatten_tensors(self, tensors, separator=4097):
        """Flatten tensors into a flat list of integers with specific popping rules."""
        flattened = []
        tensor0, tensor1, tensor2 = tensors
        queue0, queue1, queue2 = deque(tensor0.tolist()[0]), deque(tensor1.tolist()[0]), deque(tensor2.tolist()[0])

        while queue0 or queue1 or queue2:
            flattened.append(separator)
            if queue0:
                flattened.append(queue0.popleft())
            if len(queue1) >= 2:
                flattened.extend([queue1.popleft(), queue1.popleft()])
            if len(queue2) >= 4:
                flattened.extend([queue2.popleft() for _ in range(4)])
        flattened.append(separator)
        return flattened


    def save_flattened_to_text(self, flattened, path):
        with open(path, 'w') as f:
            for value in flattened:
                f.write(f"{value}\n")
        print(f"Flattened tensors saved to {path}")

    def load_flattened_from_text(self, path):
        with open(path, 'r') as f:
            flattened = [int(line.strip()) for line in f]
        return flattened

    def unflatten_tensors(self, flattened_output, separator=4097):
        """Reconstructs the list of tensors from the flattened output."""
        tensor0, tensor1, tensor2 = [], [], []
        temp_tensor = []

        for val in flattened_output:
            if val == separator:
                if temp_tensor:
                    tensor0.append(temp_tensor[0])
                    tensor1.extend(temp_tensor[1:3])
                    tensor2.extend(temp_tensor[3:7])
                    temp_tensor = []
            else:
                temp_tensor.append(val)
                if len(temp_tensor) > 7:  # Discard additional tokens beyond the required ones before the next separator
                    temp_tensor = temp_tensor[:7]

        # Convert lists to tensors and reshape appropriately
        tensor0 = torch.tensor(tensor0).view(-1, 1)
        tensor1 = torch.tensor(tensor1).view(-1, 2)
        tensor2 = torch.tensor(tensor2).view(-1, 4)

        output = [torch.flatten(tensor0).unsqueeze(0), torch.flatten(tensor1).unsqueeze(0), torch.flatten(tensor2).unsqueeze(0)]
        return output

def preprocess_audio_to_24khz(input_path, output_path, target_sample_rate=24000):
    # Load the audio file
    waveform, original_sample_rate = torchaudio.load(input_path)

    # If the audio is stereo, convert it to mono
    if waveform.shape[0] == 2:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample the audio if the original sample rate is different from the target sample rate
    if original_sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)

    # Save the preprocessed audio to the output path
    torchaudio.save(output_path, waveform, target_sample_rate)

    print(f"Audio preprocessed and saved to {output_path}")

def load_mp3_as_tensor(audio_path, sample_rate=24000):
    waveform, orig_sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] == 2:
        waveform = waveform.mean(dim=0, keepdim=True)
    if orig_sample_rate != sample_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=orig_sample_rate, new_freq=sample_rate)(waveform)
    waveform = waveform.unsqueeze(0)
    return waveform

def save_tensor_to_mp3(tensor, path, sample_rate=24000):
    tensor = tensor.squeeze(0).cpu()
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)  # Ensure at least one channel
    elif tensor.dim() == 2 and tensor.size(0) > 1:
        tensor = tensor.transpose(0, 1)  # Ensure the correct shape for torchaudio

    print(f"Saving tensor with shape: {tensor.shape}")

    torchaudio.save(path, tensor, sample_rate, format='mp3')

def encode_audio(model, audio_path, json_output_path, text_output_path):
    if audio_path.endswith('.mp3'):
        # Preprocess the audio to 24kHz and save it to a temporary file
        temp_wav_path = "temp.wav"
        preprocess_audio_to_24khz(audio_path, temp_wav_path)
        audio = load_mp3_as_tensor(temp_wav_path).to(model.device)
    else:
        audio = torch.load(audio_path).to(model.device)

    codes = model.encode(audio)
    model.save_to_json(codes, json_output_path)

    flattened = model.flatten_tensors(codes)
    model.save_flattened_to_text(flattened, text_output_path)

def decode_audio(model, input_path, output_path, input_format='json'):
    if input_format == 'json':
        codes = model.load_from_json(input_path)
    elif input_format == 'text':
        flattened = model.load_flattened_from_text(input_path)
        codes = model.unflatten_tensors(flattened)
        temp_json_path = "temp.json"
        model.save_to_json(codes, temp_json_path)
        codes = model.load_from_json(temp_json_path)

    audio_hat = model.decode(codes)
    save_tensor_to_mp3(audio_hat, output_path)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Encode and decode audio using SNAC")
    parser.add_argument('mode', choices=['encode', 'decode'], help="Mode: encode or decode")
    parser.add_argument('input', help="Input file path (.mp3 for encode, .json or .txt for decode)")
    parser.add_argument('output', help="Output file path (.json and .txt for encode, .mp3 for decode)")
    parser.add_argument('--input_format', choices=['json', 'text'], default='json', help="Input format for decoding (json or text)")
    args = parser.parse_args()

    if args.mode == 'encode':
        encode_audio(SpeechTokenizer('cuda'), args.input, args.output + '.json', args.output + '.txt')
    elif args.mode == 'decode':
        decode_audio(SpeechTokenizer('cuda'), args.input, args.output, input_format=args.input_format)

if __name__ == "__main__":
    main()

